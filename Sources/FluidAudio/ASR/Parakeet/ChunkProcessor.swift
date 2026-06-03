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
        var chunkOutputs: [[TokenWindow]] = []

        var chunkStart = 0
        var chunkIndex = 0
        var chunkDecoderState = TdtDecoderState.make(
            decoderLayers: await manager.getDecoderLayers()
        )

        while chunkStart < totalSamples {
            try Task.checkCancellation()
            let candidateEnd = chunkStart + chunkSamples
            let isLastChunk = candidateEnd >= totalSamples
            let chunkEnd = isLastChunk ? totalSamples : candidateEnd

            if chunkEnd <= chunkStart {
                break
            }

            chunkDecoderState.reset()

            // For chunks after the first, prepend context samples from the overlap region.
            // This provides left context for the mel spectrogram STFT window and encoder convolutions.
            let contextSamples = chunkIndex > 0 ? melContextSamples : 0
            let contextStart = chunkStart - contextSamples
            let chunkLengthWithContext = chunkEnd - contextStart
            let chunkSamplesArray = try readSamples(offset: contextStart, count: chunkLengthWithContext)

            let (windowTokens, windowTimestamps, windowConfidences, windowDurations) = try await transcribeChunk(
                samples: chunkSamplesArray,
                contextSamples: contextSamples,
                chunkStart: chunkStart,
                isLastChunk: isLastChunk,
                using: manager,
                decoderState: &chunkDecoderState
            )

            // Combine tokens, timestamps, and confidences into aligned tuples
            guard windowTokens.count == windowTimestamps.count && windowTokens.count == windowConfidences.count else {
                throw ASRError.processingFailed("Token, timestamp, and confidence arrays are misaligned")
            }

            // Default to 0 per token if durations array is misaligned (shouldn't happen in practice)
            let durations =
                windowDurations.count == windowTokens.count
                ? windowDurations : Array(repeating: 0, count: windowTokens.count)

            let windowData: [TokenWindow] = zip(
                zip(zip(windowTokens, windowTimestamps), windowConfidences), durations
            ).map {
                (token: $0.0.0.0, timestamp: $0.0.0.1, confidence: $0.0.1, duration: $0.1)
            }
            chunkOutputs.append(windowData)

            chunkIndex += 1

            if isLastChunk {
                break
            }

            if let progressHandler {
                let progress = min(1.0, max(0.0, Double(chunkEnd) / Double(totalSamples)))
                await progressHandler(progress)
            }

            chunkStart += strideSamples
        }

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

    private func readSamples(offset: Int, count: Int) throws -> [Float] {
        var buffer = [Float](repeating: 0, count: count)
        try buffer.withUnsafeMutableBufferPointer { pointer in
            try sampleSource.copySamples(into: pointer.baseAddress!, offset: offset, count: count)
        }
        return buffer
    }

    private func transcribeChunk(
        samples: [Float],
        contextSamples: Int,
        chunkStart: Int,
        isLastChunk: Bool,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], durations: [Int]) {
        guard !samples.isEmpty else { return ([], [], [], []) }

        let paddedChunk = manager.padAudioIfNeeded(samples, targetLength: maxModelSamples)

        // Calculate frame count for the ACTUAL audio (excluding prepended context)
        let actualAudioSamples = samples.count - contextSamples
        let actualFrameCount = ASRConstants.calculateEncoderFrames(from: actualAudioSamples)

        // Global frame offset is based on original chunkStart (not context-adjusted start)
        let globalFrameOffset = chunkStart / ASRConstants.samplesPerEncoderFrame

        // Context frame adjustment tells decoder to skip the prepended context frames
        let contextFrames = contextSamples / ASRConstants.samplesPerEncoderFrame

        let (hypothesis, encoderSequenceLength) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: samples.count,  // Full length including context
            actualAudioFrames: actualFrameCount,  // Only actual audio frames (excluding context)
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrames,  // Skip context frames in decoder
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset
        )

        if hypothesis.isEmpty || encoderSequenceLength == 0 {
            return ([], [], [], [])
        }

        return (hypothesis.ySequence, hypothesis.timestamps, hypothesis.tokenConfidences, hypothesis.tokenDurations)
    }

}
