import AVFoundation
@preconcurrency import CoreML
import Foundation
import os

/// Callback invoked when new tokens are decoded (for live transcription updates)
public typealias NemotronPartialCallback = @Sendable (String) -> Void

public struct NemotronComponentProfile: Codable, Sendable {
    public var chunks: Int = 0
    public var decodeSteps: Int = 0
    public var audioInputTime: Double = 0
    public var preprocessorTime: Double = 0
    public var melInputTime: Double = 0
    public var encoderTime: Double = 0
    public var encoderStepCopyTime: Double = 0
    public var decoderTime: Double = 0
    public var jointDecisionTime: Double = 0
    public var jointTime: Double = 0
    public var decodeLoopTime: Double = 0
    public var totalChunkTime: Double = 0

    public init() {}
}

/// High-level manager for Nemotron Speech Streaming 0.6B pipeline.
/// Implements true streaming with encoder cache states.
public actor NemotronStreamingAsrManager {
    private let logger = AppLogger(category: "NemotronStreaming")
    private let signposter = OSSignposter(subsystem: AppLogger.defaultSubsystem, category: .pointsOfInterest)

    // Models
    internal var preprocessor: MLModel?
    internal var encoder: MLModel?
    internal var encoderInit: MLModel?
    internal var encoderStep: MLModel?
    internal var decoder: MLModel?
    internal var joint: MLModel?
    internal var jointDecision: MLModel?

    // Components
    private let audioConverter = AudioConverter()
    internal var tokenizer: Tokenizer?

    // Configuration (loaded from metadata.json)
    public private(set) var config: NemotronStreamingConfig
    internal var activeTargetLanguage: String?
    internal var promptVectorCache: MLMultiArray?

    // Audio Buffer
    private var audioBuffer: [Float] = []

    // Accumulated token IDs
    internal var accumulatedTokenIds: [Int] = []

    // Encoder cache states
    internal var cacheChannel: MLMultiArray?
    internal var cacheTime: MLMultiArray?
    internal var cacheLen: MLMultiArray?
    internal var encoderState: Any?

    // Mel cache (last 9 frames from previous chunk)
    internal var melCache: MLMultiArray?

    // Decoder LSTM states
    internal var hState: MLMultiArray?
    internal var cState: MLMultiArray?
    internal var lastToken: Int32

    // Callbacks
    internal var partialCallback: NemotronPartialCallback?

    /// Chunk size for auto-download. Set by `StreamingAsrEngineFactory`
    /// to determine which HuggingFace repo to download from in `loadModels()`.
    internal var requestedChunkSize: NemotronChunkSize?

    // Stats
    internal var processedChunks: Int = 0
    internal var componentProfilingEnabled = false
    internal var componentProfile = NemotronComponentProfile()

    public private(set) var mlConfiguration: MLModelConfiguration

    public static func defaultModelConfiguration() -> MLModelConfiguration {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        return configuration
    }

    public init(
        configuration: MLModelConfiguration = NemotronStreamingAsrManager.defaultModelConfiguration(),
        requestedChunkSize: NemotronChunkSize? = nil
    ) {
        self.mlConfiguration = configuration
        self.requestedChunkSize = requestedChunkSize
        self.config = NemotronStreamingConfig()
        self.activeTargetLanguage = nil
        self.promptVectorCache = nil
        self.lastToken = Int32(config.blankIdx)
    }

    /// Set callback for partial transcription updates
    public func setPartialCallback(_ callback: @escaping NemotronPartialCallback) {
        self.partialCallback = callback
    }

    public func setComponentProfilingEnabled(_ enabled: Bool) {
        componentProfilingEnabled = enabled
        componentProfile = NemotronComponentProfile()
    }

    public func componentProfileSnapshot() -> NemotronComponentProfile {
        componentProfile
    }

    internal func profileNow() -> Double {
        Date.timeIntervalSinceReferenceDate
    }

    internal func beginProfileInterval(_ name: StaticString) -> OSSignpostIntervalState? {
        guard componentProfilingEnabled else { return nil }
        return signposter.beginInterval(name, id: signposter.makeSignpostID())
    }

    internal func endProfileInterval(_ name: StaticString, _ state: OSSignpostIntervalState?) {
        guard let state else { return }
        signposter.endInterval(name, state)
    }

    /// Select the runtime prompt language for multilingual Nemotron 3.5 bundles.
    public func setTargetLanguage(_ language: String) throws {
        guard config.runtimePrompt else {
            throw ASRError.processingFailed("Current Nemotron model does not expose a runtime prompt input")
        }
        guard config.promptDictionary[language] != nil else {
            let available = config.promptDictionary.keys.sorted().prefix(12).joined(separator: ", ")
            throw ASRError.processingFailed("Unknown Nemotron prompt language '\(language)'. Available: \(available)")
        }
        activeTargetLanguage = language
        promptVectorCache = nil
    }

    /// Load models from a directory containing preprocessor, encoder, decoder, joint, and tokenizer
    /// - Parameters:
    ///   - modelDir: Directory containing the model files
    public func loadModels(modelDir: URL) async throws {
        logger.info("Loading Nemotron CoreML models from \(modelDir.path)...")

        // Load config from metadata.json
        let metadataPath = modelDir.appendingPathComponent(ModelNames.NemotronStreaming.metadata)
        if FileManager.default.fileExists(atPath: metadataPath.path) {
            self.config = try NemotronStreamingConfig(from: metadataPath)
            self.activeTargetLanguage = config.targetLang
            self.promptVectorCache = nil
            logger.info("Loaded config: \(config.chunkMs)ms chunks, \(config.chunkMelFrames) mel frames")

            // Offline ("Accurate") encoder is GPU-optimal on Apple Silicon; ANE
            // fragments it (CoreML places ~71 ops on GPU with expensive ANE<->GPU hops).
            // Route offline to GPU; keep streaming (single/split encoder) on ANE.
            // Only override the default so explicit callers are respected.
            if config.modelLayout == .offlineEncoder,
                mlConfiguration.computeUnits == .cpuAndNeuralEngine {
                mlConfiguration.computeUnits = .cpuAndGPU
                logger.info("Nemotron offline layout -> routing encoder to cpuAndGPU (ANE-hostile graph)")
            }
        }

        // Load preprocessor
        let preprocessorPath = try resolveModelURL(
            in: modelDir,
            candidates: [ModelNames.NemotronStreaming.preprocessorFile, "preprocessor.mlpackage"]
        )
        self.preprocessor = try await loadCoreMLModel(at: preprocessorPath)

        self.encoderInit = nil
        self.encoderStep = nil
        switch config.modelLayout {
        case .singleEncoder, .offlineEncoder:
            let encoderPath = try resolveModelURL(
                in: modelDir,
                candidates: ["encoder.mlmodelc", "encoder.mlpackage"]
            )
            self.encoder = try await loadCoreMLModel(at: encoderPath)
        case .splitEncoder:
            let encoderInitPath = try resolveModelURL(
                in: modelDir,
                candidates: ["encoder_init.mlmodelc", "encoder_init.mlpackage"]
            )
            let encoderStepPath = try resolveModelURL(
                in: modelDir,
                candidates: ["encoder_step.mlmodelc", "encoder_step.mlpackage"]
            )
            self.encoderInit = try await loadCoreMLModel(at: encoderInitPath)
            self.encoderStep = try await loadCoreMLModel(at: encoderStepPath)
            self.encoder = self.encoderInit
        case .legacy:
            let encoderPath =
                modelDir
                .appendingPathComponent("encoder")
                .appendingPathComponent(NemotronEncoder.fileName)
            self.encoder = try await loadCoreMLModel(at: encoderPath)
        }

        // Load decoder
        let decoderPath = try resolveModelURL(
            in: modelDir,
            candidates: [ModelNames.NemotronStreaming.decoderFile, "decoder.mlpackage"]
        )
        self.decoder = try await loadCoreMLModel(at: decoderPath)

        if let jointPath = resolveOptionalModelURL(
            in: modelDir,
            candidates: [ModelNames.NemotronStreaming.jointFile, "joint.mlpackage"]
        ) {
            self.joint = try await loadCoreMLModel(at: jointPath)
        }

        if let jointDecisionPath = resolveOptionalModelURL(
            in: modelDir,
            candidates: ["joint_decision.mlmodelc", "joint_decision.mlpackage"]
        ) {
            self.jointDecision = try await loadCoreMLModel(at: jointDecisionPath)
        }

        guard self.joint != nil || self.jointDecision != nil else {
            throw ASRError.processingFailed("Missing model file: joint or joint_decision")
        }

        // Load tokenizer
        self.tokenizer = try Tokenizer(modelDirectory: modelDir)

        // Initialize states
        try resetStates()

        logger.info(
            "Nemotron models loaded successfully (\(config.chunkMs)ms shift, \(config.chunkMelFrames * 10)ms window).")
    }

    private func resolveModelURL(in modelDir: URL, candidates: [String]) throws -> URL {
        if let url = resolveOptionalModelURL(in: modelDir, candidates: candidates) {
            return url
        }
        throw ASRError.processingFailed("Missing model file: \(candidates.joined(separator: " or "))")
    }

    private func resolveOptionalModelURL(in modelDir: URL, candidates: [String]) -> URL? {
        for candidate in candidates {
            let url = modelDir.appendingPathComponent(candidate)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        return nil
    }

    private func loadCoreMLModel(at url: URL) async throws -> MLModel {
        let modelURL: URL
        if url.pathExtension == "mlpackage" {
            let siblingCompiled = url.deletingPathExtension().appendingPathExtension("mlmodelc")
            if FileManager.default.fileExists(atPath: siblingCompiled.path) {
                modelURL = siblingCompiled
            } else {
                modelURL = try compiledModelURL(for: url)
            }
        } else {
            modelURL = url
        }
        return try await MLModel.load(contentsOf: modelURL, configuration: mlConfiguration)
    }

    private func compiledModelURL(for packageURL: URL) throws -> URL {
        let fileManager = FileManager.default
        let cacheRoot = compiledArtifactsDirectory(for: packageURL)
        try fileManager.createDirectory(at: cacheRoot, withIntermediateDirectories: true)

        let attributes = try fileManager.attributesOfItem(atPath: packageURL.path)
        let modifiedAt = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let baseName = packageURL.deletingPathExtension().lastPathComponent
        let compiledName =
            "\(baseName)_\(computeUnitsKey(mlConfiguration.computeUnits))_\(Int64(modifiedAt * 1000)).mlmodelc"
        let compiledURL = cacheRoot.appendingPathComponent(compiledName, isDirectory: true)
        if fileManager.fileExists(atPath: compiledURL.path) {
            return compiledURL
        }

        let lockURL = cacheRoot.appendingPathComponent("\(compiledName).lock", isDirectory: true)
        while true {
            do {
                try fileManager.createDirectory(at: lockURL, withIntermediateDirectories: false)
                break
            } catch {
                if fileManager.fileExists(atPath: lockURL.path) {
                    Thread.sleep(forTimeInterval: 0.1)
                    continue
                }
                throw error
            }
        }
        defer { try? fileManager.removeItem(at: lockURL) }

        if fileManager.fileExists(atPath: compiledURL.path) {
            return compiledURL
        }

        let tempCompiledURL = try MLModel.compileModel(at: packageURL)
        let stagedURL = cacheRoot.appendingPathComponent(
            "\(compiledName).staging.\(UUID().uuidString)", isDirectory: true)
        try? fileManager.removeItem(at: stagedURL)
        try fileManager.copyItem(at: tempCompiledURL, to: stagedURL)

        if fileManager.fileExists(atPath: compiledURL.path) {
            try? fileManager.removeItem(at: stagedURL)
            try? fileManager.removeItem(at: tempCompiledURL)
            return compiledURL
        }

        try fileManager.moveItem(at: stagedURL, to: compiledURL)
        try? fileManager.removeItem(at: tempCompiledURL)
        return compiledURL
    }

    private func compiledArtifactsDirectory(for packageURL: URL) -> URL {
        let cachesDirectory =
            FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? packageURL.deletingLastPathComponent()
        return
            cachesDirectory
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("CompiledNemotronModels", isDirectory: true)
            .appendingPathComponent(stableCompiledDirectoryName(for: packageURL), isDirectory: true)
    }

    private func computeUnitsKey(_ computeUnits: MLComputeUnits) -> String {
        switch computeUnits {
        case .cpuOnly: return "cpu"
        case .cpuAndGPU: return "gpu"
        case .cpuAndNeuralEngine: return "ane"
        case .all: return "all"
        @unknown default: return "unknown"
        }
    }

    private func stableCompiledDirectoryName(for packageURL: URL) -> String {
        let path = packageURL.standardizedFileURL.path
        let folderName = packageURL.deletingPathExtension().lastPathComponent.replacingOccurrences(of: " ", with: "_")
        return "\(folderName)-\(fnv1a64(path))"
    }

    private func fnv1a64(_ string: String) -> String {
        var hash: UInt64 = 0xCBF2_9CE4_8422_2325
        let prime: UInt64 = 0x100_0000_01B3
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash &*= prime
        }
        return String(hash, radix: 16, uppercase: false)
    }

    /// Reset all states for a new transcription session
    public func reset() async {
        audioBuffer.removeAll()
        accumulatedTokenIds.removeAll()
        processedChunks = 0
        do {
            try resetStates()
        } catch {
            logger.error("Failed to reset states: \(error.localizedDescription)")
        }
    }

    private func resetStates() throws {
        let encoderCacheType: MLMultiArrayDataType = config.encoderCacheFloat16 ? .float16 : .float32

        // Encoder cache states
        if config.encoderStateful {
            guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *) else {
                throw ASRError.processingFailed("Stateful Nemotron encoder requires macOS 15/iOS 18 or newer")
            }
            cacheChannel = nil
            encoderState = encoder?.makeState()
        } else {
            encoderState = nil
            cacheChannel = try MLMultiArray(
                shape: config.cacheChannelShape.map { NSNumber(value: $0) },
                dataType: encoderCacheType
            )
            if let cacheChannel {
                zeroArray(cacheChannel)
            }
        }

        cacheTime = try MLMultiArray(
            shape: config.cacheTimeShape.map { NSNumber(value: $0) },
            dataType: encoderCacheType
        )
        if let cacheTime {
            zeroArray(cacheTime)
        }

        cacheLen = try MLMultiArray(shape: [1], dataType: .int32)
        cacheLen?[0] = 0

        // Mel cache (will be initialized on first chunk)
        melCache = nil

        // Decoder LSTM states
        hState = try MLMultiArray(
            shape: [NSNumber(value: config.decoderLayers), 1, NSNumber(value: config.decoderHidden)],
            dataType: .float32
        )
        hState?.reset(to: 0)

        cState = try MLMultiArray(
            shape: [NSNumber(value: config.decoderLayers), 1, NSNumber(value: config.decoderHidden)],
            dataType: .float32
        )
        cState?.reset(to: 0)

        lastToken = Int32(config.blankIdx)
    }

    private nonisolated func zeroArray(_ array: MLMultiArray) {
        let bytesPerElement: Int
        switch array.dataType {
        case .float16:
            bytesPerElement = MemoryLayout<UInt16>.stride
        case .float32, .int32:
            bytesPerElement = MemoryLayout<UInt32>.stride
        case .double:
            bytesPerElement = MemoryLayout<Double>.stride
        @unknown default:
            bytesPerElement = MemoryLayout<Float>.stride
        }
        memset(array.dataPointer, 0, array.count * bytesPerElement)
    }

    /// Append audio buffer for processing
    public func appendAudio(_ buffer: AVAudioPCMBuffer) throws {
        let samples = try audioConverter.resampleBuffer(buffer)
        audioBuffer.append(contentsOf: samples)
    }

    /// Process audio and return partial transcript
    public func process(audioBuffer: AVAudioPCMBuffer) async throws -> String {
        // Check if models are loaded
        try ensureModelsLoaded()

        let samples = try audioConverter.resampleBuffer(audioBuffer)
        self.audioBuffer.append(contentsOf: samples)

        if config.modelLayout == .splitEncoder {
            return ""
        }

        // Process complete chunks
        while self.audioBuffer.count >= config.chunkSamples {
            let chunk = Array(self.audioBuffer.prefix(config.chunkSamples))
            try await processChunk(chunk)
            // Recheck buffer count after await to handle actor reentrancy
            let samplesToRemove = min(config.shiftSamples, self.audioBuffer.count)
            self.audioBuffer.removeFirst(samplesToRemove)
        }

        return ""
    }

    /// Transcribe a complete buffer through the same streaming chunk path without mutating
    /// the incremental audio queue. This keeps benchmark runs from spending time shifting
    /// large Swift arrays while preserving the model/cache contract used for streaming.
    public func transcribe(audioBuffer: AVAudioPCMBuffer) async throws -> String {
        guard let tokenizer = tokenizer else {
            throw ASRError.notInitialized
        }
        try ensureModelsLoaded()

        self.audioBuffer.removeAll()
        self.accumulatedTokenIds.removeAll()
        self.processedChunks = 0
        try resetStates()

        let samples = try audioConverter.resampleBuffer(audioBuffer)
        guard !samples.isEmpty else {
            return ""
        }

        if config.modelLayout == .offlineEncoder {
            try await transcribeOffline(samples: samples)
        } else if config.modelLayout == .splitEncoder {
            try await transcribeSplitEncoder(samples: samples)
        } else {
            var offset = 0
            while offset < samples.count {
                let end = min(offset + config.chunkSamples, samples.count)
                var chunk = Array(samples[offset..<end])
                if chunk.count < config.chunkSamples {
                    chunk.append(contentsOf: repeatElement(0.0, count: config.chunkSamples - chunk.count))
                }
                try await processChunk(chunk)
                offset += config.shiftSamples
            }
        }

        let transcript = tokenizer.decode(
            ids: accumulatedTokenIds,
            skipSpecialTokens: config.modelLayout != .legacy
        )
        accumulatedTokenIds.removeAll()
        return transcript
    }

    /// Finish processing and return final transcript
    public func finish() async throws -> String {
        // Check if models are loaded
        guard let tokenizer = tokenizer else {
            throw ASRError.notInitialized
        }
        try ensureModelsLoaded()

        if !audioBuffer.isEmpty {
            if config.modelLayout == .splitEncoder {
                accumulatedTokenIds.removeAll()
                processedChunks = 0
                try resetStates()
                try await transcribeSplitEncoder(samples: audioBuffer)
            } else {
                let paddingNeeded = config.chunkSamples - audioBuffer.count
                if paddingNeeded > 0 {
                    audioBuffer.append(contentsOf: Array(repeating: 0.0, count: paddingNeeded))
                }

                let chunk = Array(audioBuffer.prefix(config.chunkSamples))
                try await processChunk(chunk)
            }
            audioBuffer.removeAll()
        }

        // Decode accumulated tokens
        let transcript = tokenizer.decode(
            ids: accumulatedTokenIds,
            skipSpecialTokens: config.modelLayout != .legacy
        )
        accumulatedTokenIds.removeAll()

        return transcript
    }

    /// Offline full-context encoder.
    ///
    /// The offline encoder is converted to a fixed CoreML window (`config.maxMelFrames`), so a
    /// single pass can only cover ~one window of audio (and a single pass over very long audio
    /// would exhaust memory). For longer audio we slide that window with overlap and merge the
    /// per-window token streams with `AsrChunkTokenMerger` — the same overlap-merge the Parakeet
    /// `ChunkProcessor` uses. The public `transcribe`/`finish` API is unchanged.
    internal func transcribeOffline(samples: [Float]) async throws {
        guard preprocessor != nil, encoder != nil else {
            throw ASRError.notInitialized
        }

        let sampleRate = 16000
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        let melCap =
            config.maxMelFrames > 0
            ? config.maxMelFrames
            : (ASRConstants.maxModelSamples / ASRConstants.melHopSize)
        let maxWindowSamples = melCap * ASRConstants.melHopSize
        // Frame-aligned window, leaving one hop of slack so a full window's mel stays within the
        // fixed encoder width.
        let rawWindow = max(maxWindowSamples - ASRConstants.melHopSize, frameSamples)
        let windowSamples = (rawWindow / frameSamples) * frameSamples
        let overlapRequested = Int(2.0 * Double(sampleRate))
        let overlapCapped = min(overlapRequested, windowSamples / 2)
        let overlapSamples = (overlapCapped / frameSamples) * frameSamples
        let strideSamples = max(windowSamples - overlapSamples, frameSamples)

        // Whole utterance fits one encoder window → single pass (prior behavior).
        if samples.count <= windowSamples {
            try await decodeOfflineWindow(samples: samples, tokenSink: nil)
            processedChunks += 1
            return
        }

        // Long audio → sliding window. Each window is an independent full-context pass with a
        // fresh RNNT decoder state; overlaps are reconciled by the shared token merger.
        var chunkOutputs: [[AsrChunkTokenMerger.TokenWindow]] = []
        var start = 0
        while start < samples.count {
            try Task.checkCancellation()
            let end = min(start + windowSamples, samples.count)
            let windowAudio = Array(samples[start..<end])
            let startFrame = start / frameSamples
            var windowTokens: [AsrChunkTokenMerger.TokenWindow] = []
            try await decodeOfflineWindow(samples: windowAudio) { tokenId, frame in
                windowTokens.append((token: tokenId, timestamp: startFrame + frame, confidence: 1.0, duration: 0))
            }
            chunkOutputs.append(windowTokens)
            if end >= samples.count { break }
            start += strideSamples
        }

        let merged = AsrChunkTokenMerger.merge(chunkOutputs, sampleRate: sampleRate, overlapSeconds: 2.0)
        accumulatedTokenIds = merged.map { $0.token }
        processedChunks += chunkOutputs.count
    }

    /// Run the offline encoder + RNNT greedy decode over a single window of audio.
    /// `tokenSink` nil → emitted tokens append to `accumulatedTokenIds` (single-pass path).
    /// `tokenSink` set → emitted `(tokenId, encoderFrame)` are reported for window merging.
    private func decodeOfflineWindow(
        samples: [Float],
        tokenSink: ((Int, Int) -> Void)?
    ) async throws {
        guard let preprocessor, let encoder else {
            throw ASRError.notInitialized
        }

        let audioArray = try createAudioArray(samples)
        let audioLen = try MLMultiArray(shape: [1], dataType: .int32)
        audioLen[0] = NSNumber(value: samples.count)
        let preprocInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audioArray),
            "audio_length": MLFeatureValue(multiArray: audioLen),
        ])
        let preprocOutput = try await preprocessor.prediction(from: preprocInput)
        guard let processedSignal = preprocOutput.featureValue(for: "processed_signal")?.multiArrayValue else {
            throw ASRError.processingFailed("Preprocessor failed to produce processed_signal output")
        }
        let validFrames: Int
        if let lengthArray = preprocOutput.featureValue(for: "processed_signal_length")?.multiArrayValue {
            validFrames = max(0, lengthArray[0].intValue)
        } else {
            validFrames = processedSignal.shape[2].intValue
        }

        let melWidth = config.maxMelFrames > 0 ? config.maxMelFrames : processedSignal.shape[2].intValue
        let copyFrames = min(validFrames, melWidth)
        let encoderMel = try sliceFullMel(from: processedSignal, copyFrames: copyFrames, targetWidth: melWidth)
        let melLen = try MLMultiArray(shape: [1], dataType: .int32)
        melLen[0] = NSNumber(value: copyFrames)

        var feats: [String: MLFeatureValue] = [
            "processed_signal": MLFeatureValue(multiArray: encoderMel),
            "processed_signal_length": MLFeatureValue(multiArray: melLen),
        ]
        if config.runtimePrompt {
            feats["prompt_vector"] = MLFeatureValue(multiArray: try createPromptVector())
        }
        let encoderOutput = try await encoder.prediction(from: try MLDictionaryFeatureProvider(dictionary: feats))
        guard let encoded = encoderOutput.featureValue(for: "encoded")?.multiArrayValue else {
            throw ASRError.processingFailed("Offline encoder failed to produce encoded output")
        }
        let numEncoderFrames = validEncoderFrameCount(from: encoderOutput, encoded: encoded)

        var currentToken = Int32(config.blankIdx)
        var currentH = try makeZeroDecoderState()
        var currentC = try makeZeroDecoderState()
        try await runGreedyRnntDecodeLoop(
            encoded: encoded,
            numEncoderFrames: numEncoderFrames,
            currentToken: &currentToken,
            currentH: &currentH,
            currentC: &currentC,
            shouldProfile: false,
            tokenSink: tokenSink
        )
    }

    private func makeZeroDecoderState() throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [NSNumber(value: config.decoderLayers), 1, NSNumber(value: config.decoderHidden)],
            dataType: .float32
        )
        array.reset(to: 0)
        return array
    }

    private func sliceFullMel(from source: MLMultiArray, copyFrames: Int, targetWidth: Int) throws -> MLMultiArray {
        let melFeatures = config.melFeatures
        let result = try MLMultiArray(
            shape: [1, NSNumber(value: melFeatures), NSNumber(value: targetWidth)],
            dataType: .float32
        )
        result.reset(to: 0)
        guard copyFrames > 0 else { return result }
        let srcPtr = source.dataPointer.bindMemory(to: Float.self, capacity: source.count)
        let dstPtr = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)
        let s0 = source.strides[0].intValue
        let s1 = source.strides[1].intValue
        let s2 = source.strides[2].intValue
        let d0 = result.strides[0].intValue
        let d1 = result.strides[1].intValue
        let d2 = result.strides[2].intValue
        for mel in 0..<melFeatures {
            for t in 0..<copyFrames {
                dstPtr[0 * d0 + mel * d1 + t * d2] = srcPtr[0 * s0 + mel * s1 + t * s2]
            }
        }
        return result
    }

    /// NeMo cache-aware mel chunking for split `encoder_init` / `encoder_step` bundles.
    internal func transcribeSplitEncoder(samples: [Float]) async throws {
        guard let preprocessor else {
            throw ASRError.notInitialized
        }

        let audioArray = try createAudioArray(samples)
        let audioLen = try MLMultiArray(shape: [1], dataType: .int32)
        audioLen[0] = NSNumber(value: samples.count)

        let preprocInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audioArray),
            "audio_length": MLFeatureValue(multiArray: audioLen),
        ])
        let preprocOutput = try await preprocessor.prediction(from: preprocInput)
        guard let processedSignal = preprocOutput.featureValue(for: "processed_signal")?.multiArrayValue else {
            throw ASRError.processingFailed("Preprocessor failed to produce processed_signal output")
        }

        let validFrames: Int
        if let lengthArray = preprocOutput.featureValue(for: "processed_signal_length")?.multiArrayValue {
            validFrames = max(0, lengthArray[0].intValue)
        } else {
            validFrames = processedSignal.shape[2].intValue
        }

        let melChunks = try NemotronCacheAwareMelChunker.makeChunks(
            from: processedSignal,
            validFrames: validFrames,
            config: config
        )
        for melChunk in melChunks {
            try await processSplitEncoderMelChunk(melChunk)
        }
    }

    private func ensureModelsLoaded() throws {
        let hasEncoder = encoder != nil || (encoderInit != nil && encoderStep != nil)
        guard preprocessor != nil, hasEncoder, decoder != nil, joint != nil || jointDecision != nil else {
            throw ASRError.notInitialized
        }
    }

    /// Get current partial transcript without finishing
    public func getPartialTranscript() -> String {
        guard let tokenizer = tokenizer else { return "" }
        return tokenizer.decode(
            ids: accumulatedTokenIds,
            skipSpecialTokens: config.modelLayout != .legacy
        )
    }
}

// MARK: - StreamingAsrEngine Conformance

extension NemotronStreamingAsrManager: StreamingAsrEngine {
    public var displayName: String {
        "Nemotron 0.6B (\(config.chunkMs)ms)"
    }

    public func loadModels() async throws {
        let chunkSize = requestedChunkSize ?? .ms1120
        let repo = chunkSize.repo

        let modelsBaseDir = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        .appendingPathComponent("FluidAudio", isDirectory: true)
        .appendingPathComponent("Models", isDirectory: true)

        let cacheDir = modelsBaseDir.appendingPathComponent(repo.folderName)
        let encoderInt8Path = cacheDir.appendingPathComponent("encoder/\(NemotronEncoder.fileName)")

        if !FileManager.default.fileExists(atPath: encoderInt8Path.path) {
            try await DownloadUtils.downloadRepo(repo, to: modelsBaseDir)
        }

        try await loadModels(modelDir: cacheDir)
    }

    public func processBufferedAudio() async throws {
        guard preprocessor != nil, encoder != nil, decoder != nil, joint != nil || jointDecision != nil else {
            throw ASRError.notInitialized
        }

        while audioBuffer.count >= config.chunkSamples {
            let chunk = Array(audioBuffer.prefix(config.chunkSamples))
            try await processChunk(chunk)
            let samplesToRemove = min(config.shiftSamples, audioBuffer.count)
            audioBuffer.removeFirst(samplesToRemove)
        }
    }

    public func setPartialTranscriptCallback(_ callback: @escaping @Sendable (String) -> Void) {
        self.partialCallback = callback
    }
}
