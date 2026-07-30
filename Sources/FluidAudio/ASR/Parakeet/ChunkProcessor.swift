import CoreML
import Foundation
import OSLog

struct ChunkProcessor {
    let sampleSource: StreamingAudioSampleSource
    let totalSamples: Int

    private let logger = AppLogger(category: "ChunkProcessor")
    private typealias TokenWindow = AsrChunkTokenMerger.TokenWindow

    // Stateless chunking aligned with CoreML reference:
    // - process ~14.96s of audio per window (frame-aligned) to stay under encoder limit
    // - 2.0s overlap (frame-aligned) to give the decoder slack when merging windows
    private let sampleRate: Int = 16000
    private let overlapSeconds: Double = 2.0

    /// Context samples prepended from previous chunk for mel spectrogram stability (80ms = 1 encoder frame).
    /// The FastConformer encoder's depthwise convolutions need left context for stable output.
    /// Without this, the first frames of a chunk may produce features that cause all-blank predictions.
    private let melContextSamples: Int = ASRConstants.samplesPerEncoderFrame  // 1280 samples = 80ms

    private var maxModelSamples: Int { ASRConstants.maxModelSamples }

    private var chunkSamples: Int {
        // Reserve space for context samples that will be prepended to non-first chunks.
        // This ensures chunkSamples + melContextSamples <= maxModelSamples.
        let maxActualChunk = maxModelSamples - melContextSamples  // 240000 - 1280 = 238720
        let raw = max(maxActualChunk - ASRConstants.melHopSize, ASRConstants.samplesPerEncoderFrame)
        return raw / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }
    private var overlapSamples: Int {
        let requested = Int(overlapSeconds * Double(sampleRate))
        let capped = min(requested, chunkSamples / 2)
        return capped / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }
    private var strideSamples: Int {
        let raw = max(chunkSamples - overlapSamples, ASRConstants.samplesPerEncoderFrame)
        return raw / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    /// Initialize with a streaming audio sample source for memory-efficient processing.
    init(sampleSource: StreamingAudioSampleSource) {
        self.sampleSource = sampleSource
        self.totalSamples = sampleSource.sampleCount
    }

    /// Convenience initializer for in-memory audio samples.
    init(audioSamples: [Float]) {
        self.init(sampleSource: ArrayAudioSampleSource(samples: audioSamples))
    }

    func process(
        using manager: AsrManager,
        startTime: Date,
        progressHandler: ((Double) async -> Void)? = nil
    ) async throws -> ASRResult {
        let pipelineMode = ProcessInfo.processInfo.environment["FLUIDAUDIO_FRONTEND_PIPELINE_MODE"]
        let supportsPipelining = await manager.supportsParakeetFrontendPipelining()
        let pipeliningEnabled = Self.shouldPipelineFrontend(
            environmentMode: pipelineMode,
            supportsPipelining: supportsPipelining
        )
        let chunkOutputs = try await processChunks(
            using: manager,
            pipeliningEnabled: pipeliningEnabled,
            progressHandler: progressHandler
        )

        guard var mergedTokens = chunkOutputs.first else {
            return await manager.processTranscriptionResult(
                tokenIds: [],
                timestamps: [],
                confidences: [],
                encoderSequenceLength: 0,
                audioSamples: [],
                processingTime: Date().timeIntervalSince(startTime)
            )
        }

        if chunkOutputs.count > 1 {
            for chunk in chunkOutputs.dropFirst() {
                mergedTokens = AsrChunkTokenMerger.mergePair(
                    mergedTokens, chunk, sampleRate: sampleRate, overlapSeconds: overlapSeconds)
            }
        }

        if mergedTokens.count > 1 {
            mergedTokens.sort { $0.timestamp < $1.timestamp }
        }

        let allTokens = mergedTokens.map { $0.token }
        let allTimestamps = mergedTokens.map { $0.timestamp }
        let allConfidences = mergedTokens.map { $0.confidence }
        let allDurations = mergedTokens.map { $0.duration }

        return await manager.processTranscriptionResult(
            tokenIds: allTokens,
            timestamps: allTimestamps,
            confidences: allConfidences,
            tokenDurations: allDurations,
            encoderSequenceLength: 0,  // Not relevant for chunk processing
            audioSamples: [],
            processingTime: Date().timeIntervalSince(startTime)
        )
    }

    static func shouldPipelineFrontend(
        environmentMode: String?,
        supportsPipelining: Bool
    ) -> Bool {
        environmentMode != "serial" && supportsPipelining
    }

    private struct ChunkWork: Sendable {
        let samples: [Float]
        let paddedSamples: [Float]
        let contextSamples: Int
        let chunkStart: Int
        let chunkEnd: Int
        let isLastChunk: Bool
    }

    private func processChunks(
        using manager: AsrManager,
        pipeliningEnabled: Bool,
        progressHandler: ((Double) async -> Void)?
    ) async throws -> [[TokenWindow]] {
        if pipeliningEnabled {
            return try await processChunksPipelined(
                using: manager,
                progressHandler: progressHandler
            )
        }
        return try await processChunksSerial(
            using: manager,
            progressHandler: progressHandler
        )
    }

    private func processChunksSerial(
        using manager: AsrManager,
        progressHandler: ((Double) async -> Void)?
    ) async throws -> [[TokenWindow]] {
        guard var currentWork = try makeChunkWork(chunkStart: 0, chunkIndex: 0) else {
            return []
        }

        var chunkOutputs: [[TokenWindow]] = []
        var chunkIndex = 0
        var chunkDecoderState = TdtDecoderState.make(
            decoderLayers: await manager.getDecoderLayers()
        )

        while true {
            var preparedPreprocessor: PreparedParakeetPreprocessorHandle?
            var preparedEncoder: PreparedParakeetEncoderHandle?
            do {
                try Task.checkCancellation()
                preparedPreprocessor = try await manager.prepareParakeetPreprocessorOutput(
                    currentWork.paddedSamples,
                    originalLength: currentWork.samples.count
                )
                guard let preprocessorToEncode = preparedPreprocessor else {
                    throw ASRError.processingFailed("Preprocessor output was not prepared")
                }
                preparedEncoder = try await manager.prepareParakeetEncoderOutput(
                    preparedPreprocessor: preprocessorToEncode
                )
                preparedPreprocessor = nil
                guard let encoderToDecode = preparedEncoder else {
                    throw ASRError.processingFailed("Encoder output was not prepared")
                }
                chunkDecoderState.reset()
                let output = try await transcribeChunk(
                    work: currentWork,
                    preparedEncoder: encoderToDecode,
                    using: manager,
                    decoderState: &chunkDecoderState
                )
                preparedEncoder = nil
                chunkOutputs.append(try makeTokenWindow(from: output))
            } catch {
                if let preparedPreprocessor {
                    await manager.discardParakeetPreprocessorOutput(preparedPreprocessor)
                }
                if let preparedEncoder {
                    await manager.discardParakeetEncoderOutput(preparedEncoder)
                }
                throw error
            }

            if currentWork.isLastChunk {
                break
            }

            if let progressHandler {
                let progress = min(1.0, max(0.0, Double(currentWork.chunkEnd) / Double(totalSamples)))
                await progressHandler(progress)
            }

            guard
                let nextWork = try makeChunkWork(
                    chunkStart: currentWork.chunkStart + strideSamples,
                    chunkIndex: chunkIndex + 1
                )
            else {
                break
            }
            currentWork = nextWork
            chunkIndex += 1
        }

        return chunkOutputs
    }

    private func processChunksPipelined(
        using manager: AsrManager,
        progressHandler: ((Double) async -> Void)?
    ) async throws -> [[TokenWindow]] {
        guard var currentWork = try makeChunkWork(chunkStart: 0, chunkIndex: 0) else {
            return []
        }

        var chunkOutputs: [[TokenWindow]] = []
        var chunkIndex = 0
        var chunkDecoderState = TdtDecoderState.make(
            decoderLayers: await manager.getDecoderLayers()
        )
        var currentEncoderTask: Task<PreparedParakeetEncoderHandle, Error>?
        var lookaheadWork: ChunkWork?
        var lookaheadPreprocessorTask: Task<PreparedParakeetPreprocessorHandle, Error>?

        while true {
            var currentEncoder: PreparedParakeetEncoderHandle?
            var initialPreprocessor: PreparedParakeetPreprocessorHandle?
            var lookaheadPreprocessor: PreparedParakeetPreprocessorHandle?
            var followingWork: ChunkWork?
            var followingPreprocessorTask: Task<PreparedParakeetPreprocessorHandle, Error>?
            var nextEncoderTask: Task<PreparedParakeetEncoderHandle, Error>?

            do {
                try Task.checkCancellation()
                if let currentEncoderTask {
                    currentEncoder = try await currentEncoderTask.value
                } else {
                    initialPreprocessor = try await manager.prepareParakeetPreprocessorOutput(
                        currentWork.paddedSamples,
                        originalLength: currentWork.samples.count
                    )
                    if !currentWork.isLastChunk {
                        lookaheadWork = try makeChunkWork(
                            chunkStart: currentWork.chunkStart + strideSamples,
                            chunkIndex: chunkIndex + 1
                        )
                        lookaheadPreprocessorTask =
                            lookaheadWork.map { makePreprocessorTask(for: $0, using: manager) }
                    }
                    guard let initialPreprocessorHandle = initialPreprocessor else {
                        throw ASRError.processingFailed("Initial preprocessor output was not prepared")
                    }
                    currentEncoder = try await manager.prepareParakeetEncoderOutput(
                        preparedPreprocessor: initialPreprocessorHandle
                    )
                    initialPreprocessor = nil
                }

                if let lookaheadWork {
                    guard let lookaheadPreprocessorTask else {
                        throw ASRError.processingFailed("Lookahead preprocessor task is unavailable")
                    }
                    lookaheadPreprocessor = try await lookaheadPreprocessorTask.value

                    if !lookaheadWork.isLastChunk {
                        followingWork = try makeChunkWork(
                            chunkStart: lookaheadWork.chunkStart + strideSamples,
                            chunkIndex: chunkIndex + 2
                        )
                        followingPreprocessorTask =
                            followingWork.map { makePreprocessorTask(for: $0, using: manager) }
                    }

                    guard let lookaheadPreprocessorHandle = lookaheadPreprocessor else {
                        throw ASRError.processingFailed("Lookahead preprocessor output was not prepared")
                    }
                    nextEncoderTask = makeEncoderTask(
                        for: lookaheadPreprocessorHandle,
                        using: manager
                    )
                    lookaheadPreprocessor = nil
                }

                guard let encoderToDecode = currentEncoder else {
                    throw ASRError.processingFailed("Encoder output was not prepared")
                }
                chunkDecoderState.reset()
                let output = try await transcribeChunk(
                    work: currentWork,
                    preparedEncoder: encoderToDecode,
                    using: manager,
                    decoderState: &chunkDecoderState
                )
                currentEncoder = nil
                chunkOutputs.append(try makeTokenWindow(from: output))
            } catch {
                if let initialPreprocessor {
                    await manager.discardParakeetPreprocessorOutput(initialPreprocessor)
                }
                if let lookaheadPreprocessor {
                    await manager.discardParakeetPreprocessorOutput(lookaheadPreprocessor)
                }
                if let currentEncoder {
                    await manager.discardParakeetEncoderOutput(currentEncoder)
                }
                await discardPreprocessorTask(lookaheadPreprocessorTask, using: manager)
                await discardPreprocessorTask(followingPreprocessorTask, using: manager)
                await discardEncoderTask(currentEncoderTask, using: manager)
                await discardEncoderTask(nextEncoderTask, using: manager)
                throw error
            }

            if currentWork.isLastChunk {
                break
            }

            if let progressHandler {
                let progress = min(1.0, max(0.0, Double(currentWork.chunkEnd) / Double(totalSamples)))
                await progressHandler(progress)
            }

            guard let nextWork = lookaheadWork else {
                break
            }
            currentWork = nextWork
            currentEncoderTask = nextEncoderTask
            lookaheadWork = followingWork
            lookaheadPreprocessorTask = followingPreprocessorTask
            chunkIndex += 1
        }

        return chunkOutputs
    }

    private func makeChunkWork(chunkStart: Int, chunkIndex: Int) throws -> ChunkWork? {
        guard chunkStart < totalSamples else { return nil }

        let candidateEnd = chunkStart + chunkSamples
        let isLastChunk = candidateEnd >= totalSamples
        let chunkEnd = isLastChunk ? totalSamples : candidateEnd
        guard chunkEnd > chunkStart else { return nil }

        let contextSamples = chunkIndex > 0 ? melContextSamples : 0
        let contextStart = chunkStart - contextSamples
        let chunkLengthWithContext = chunkEnd - contextStart
        let samples = try readSamples(offset: contextStart, count: chunkLengthWithContext)
        let paddedSamples =
            samples.count < maxModelSamples
            ? samples + Array(repeating: 0, count: maxModelSamples - samples.count)
            : samples

        return ChunkWork(
            samples: samples,
            paddedSamples: paddedSamples,
            contextSamples: contextSamples,
            chunkStart: chunkStart,
            chunkEnd: chunkEnd,
            isLastChunk: isLastChunk
        )
    }

    private func makePreprocessorTask(
        for work: ChunkWork,
        using manager: AsrManager
    ) -> Task<PreparedParakeetPreprocessorHandle, Error> {
        Task {
            try await manager.prepareParakeetPreprocessorOutput(
                work.paddedSamples,
                originalLength: work.samples.count
            )
        }
    }

    private func discardPreprocessorTask(
        _ task: Task<PreparedParakeetPreprocessorHandle, Error>?,
        using manager: AsrManager
    ) async {
        guard let task else { return }
        task.cancel()
        if let handle = try? await task.value {
            await manager.discardParakeetPreprocessorOutput(handle)
        }
    }

    private func makeEncoderTask(
        for preparedPreprocessor: PreparedParakeetPreprocessorHandle,
        using manager: AsrManager
    ) -> Task<PreparedParakeetEncoderHandle, Error> {
        Task {
            await Task.yield()
            return try await manager.prepareParakeetEncoderOutput(
                preparedPreprocessor: preparedPreprocessor
            )
        }
    }

    private func discardEncoderTask(
        _ task: Task<PreparedParakeetEncoderHandle, Error>?,
        using manager: AsrManager
    ) async {
        guard let task else { return }
        task.cancel()
        if let handle = try? await task.value {
            await manager.discardParakeetEncoderOutput(handle)
        }
    }

    private func makeTokenWindow(
        from output: (tokens: [Int], timestamps: [Int], confidences: [Float], durations: [Int])
    ) throws -> [TokenWindow] {
        guard output.tokens.count == output.timestamps.count,
            output.tokens.count == output.confidences.count
        else {
            throw ASRError.processingFailed("Token, timestamp, and confidence arrays are misaligned")
        }

        let durations =
            output.durations.count == output.tokens.count
            ? output.durations : Array(repeating: 0, count: output.tokens.count)
        return zip(
            zip(zip(output.tokens, output.timestamps), output.confidences), durations
        ).map {
            (token: $0.0.0.0, timestamp: $0.0.0.1, confidence: $0.0.1, duration: $0.1)
        }
    }

    private func readSamples(offset: Int, count: Int) throws -> [Float] {
        var buffer = [Float](repeating: 0, count: count)
        try buffer.withUnsafeMutableBufferPointer { pointer in
            try sampleSource.copySamples(into: pointer.baseAddress!, offset: offset, count: count)
        }
        return buffer
    }

    private func transcribeChunk(
        work: ChunkWork,
        preparedEncoder: PreparedParakeetEncoderHandle,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], durations: [Int]) {
        let actualAudioSamples = work.samples.count - work.contextSamples
        let actualFrameCount = ASRConstants.calculateEncoderFrames(from: actualAudioSamples)
        let globalFrameOffset = work.chunkStart / ASRConstants.samplesPerEncoderFrame
        let contextFrames = work.contextSamples / ASRConstants.samplesPerEncoderFrame

        let (hypothesis, encoderSequenceLength) = try await manager.executeMLInferenceWithTimings(
            preparedEncoder: preparedEncoder,
            paddedAudio: work.paddedSamples,
            originalLength: work.samples.count,
            actualAudioFrames: actualFrameCount,
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrames,
            isLastChunk: work.isLastChunk,
            globalFrameOffset: globalFrameOffset
        )

        if hypothesis.isEmpty || encoderSequenceLength == 0 {
            return ([], [], [], [])
        }

        return (hypothesis.ySequence, hypothesis.timestamps, hypothesis.tokenConfidences, hypothesis.tokenDurations)
    }

}
