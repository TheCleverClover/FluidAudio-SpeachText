import AVFoundation
@preconcurrency import CoreML
import Foundation

/// Callback invoked when new tokens are decoded (for live transcription updates)
public typealias NemotronPartialCallback = @Sendable (String) -> Void

/// High-level manager for Nemotron Speech Streaming 0.6B pipeline.
/// Implements true streaming with encoder cache states.
public actor NemotronStreamingAsrManager {
    private let logger = AppLogger(category: "NemotronStreaming")

    // Models
    internal var preprocessor: MLModel?
    internal var encoder: MLModel?
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

    public private(set) var mlConfiguration: MLModelConfiguration

    public init(
        configuration: MLModelConfiguration = MLModelConfiguration(),
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
        }

        // Load preprocessor
        let preprocessorPath = try resolveModelURL(
            in: modelDir,
            candidates: [ModelNames.NemotronStreaming.preprocessorFile, "preprocessor.mlpackage"]
        )
        self.preprocessor = try await loadCoreMLModel(at: preprocessorPath)

        let encoderPath: URL
        if config.modelLayout == .singleEncoder {
            encoderPath = try resolveModelURL(in: modelDir, candidates: ["encoder.mlmodelc", "encoder.mlpackage"])
        } else {
            encoderPath = modelDir.appendingPathComponent("encoder").appendingPathComponent(NemotronEncoder.fileName)
        }
        self.encoder = try await loadCoreMLModel(at: encoderPath)

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

        logger.info("Nemotron models loaded successfully (\(config.chunkMs)ms shift, \(config.chunkMelFrames * 10)ms window).")
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
        let compiledName = "\(baseName)_\(computeUnitsKey(mlConfiguration.computeUnits))_\(Int64(modifiedAt * 1000)).mlmodelc"
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
        let stagedURL = cacheRoot.appendingPathComponent("\(compiledName).staging.\(UUID().uuidString)", isDirectory: true)
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
        let cachesDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? packageURL.deletingLastPathComponent()
        return cachesDirectory
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
        // Encoder cache states
        cacheChannel = try MLMultiArray(
            shape: config.cacheChannelShape.map { NSNumber(value: $0) },
            dataType: .float32
        )
        cacheChannel?.reset(to: 0)

        cacheTime = try MLMultiArray(
            shape: config.cacheTimeShape.map { NSNumber(value: $0) },
            dataType: .float32
        )
        cacheTime?.reset(to: 0)

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

    /// Append audio buffer for processing
    public func appendAudio(_ buffer: AVAudioPCMBuffer) throws {
        let samples = try audioConverter.resampleBuffer(buffer)
        audioBuffer.append(contentsOf: samples)
    }

    /// Process audio and return partial transcript
    public func process(audioBuffer: AVAudioPCMBuffer) async throws -> String {
        // Check if models are loaded
        guard preprocessor != nil, encoder != nil, decoder != nil, (joint != nil || jointDecision != nil) else {
            throw ASRError.notInitialized
        }

        let samples = try audioConverter.resampleBuffer(audioBuffer)
        self.audioBuffer.append(contentsOf: samples)

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
        guard let tokenizer = tokenizer,
            preprocessor != nil,
            encoder != nil,
            decoder != nil,
            (joint != nil || jointDecision != nil)
        else {
            throw ASRError.notInitialized
        }

        self.audioBuffer.removeAll()
        self.accumulatedTokenIds.removeAll()
        self.processedChunks = 0
        try resetStates()

        let samples = try audioConverter.resampleBuffer(audioBuffer)
        guard !samples.isEmpty else {
            return ""
        }

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

        let transcript = tokenizer.decode(
            ids: accumulatedTokenIds,
            skipSpecialTokens: config.modelLayout == .singleEncoder
        )
        accumulatedTokenIds.removeAll()
        return transcript
    }

    /// Finish processing and return final transcript
    public func finish() async throws -> String {
        // Check if models are loaded
        guard let tokenizer = tokenizer,
            preprocessor != nil,
            encoder != nil,
            decoder != nil,
            (joint != nil || jointDecision != nil)
        else {
            throw ASRError.notInitialized
        }

        // Process remaining audio (padded if needed)
        if !audioBuffer.isEmpty {
            let paddingNeeded = config.chunkSamples - audioBuffer.count
            if paddingNeeded > 0 {
                audioBuffer.append(contentsOf: Array(repeating: 0.0, count: paddingNeeded))
            }

            let chunk = Array(audioBuffer.prefix(config.chunkSamples))
            try await processChunk(chunk)
            audioBuffer.removeAll()
        }

        // Decode accumulated tokens
        let transcript = tokenizer.decode(
            ids: accumulatedTokenIds,
            skipSpecialTokens: config.modelLayout == .singleEncoder
        )
        accumulatedTokenIds.removeAll()

        return transcript
    }

    /// Get current partial transcript without finishing
    public func getPartialTranscript() -> String {
        guard let tokenizer = tokenizer else { return "" }
        return tokenizer.decode(
            ids: accumulatedTokenIds,
            skipSpecialTokens: config.modelLayout == .singleEncoder
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
        guard preprocessor != nil, encoder != nil, decoder != nil, (joint != nil || jointDecision != nil) else {
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
