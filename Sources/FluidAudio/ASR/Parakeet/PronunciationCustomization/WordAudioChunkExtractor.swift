import Foundation

/// A word reconstructed from one or more ASR token timings.
public struct TimedTranscriptWord: Sendable, Equatable {
    public let text: String
    public let startTime: TimeInterval
    public let endTime: TimeInterval
    public let confidence: Float

    public init(text: String, startTime: TimeInterval, endTime: TimeInterval, confidence: Float) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
        self.confidence = confidence
    }
}

/// A selected transcript word and its corresponding range in the original 16 kHz audio samples.
public struct WordAudioChunk: Sendable {
    public let word: TimedTranscriptWord
    public let wordIndex: Int
    public let occurrence: Int
    public let extractionStartTime: TimeInterval
    public let extractionEndTime: TimeInterval
    public let sampleRange: Range<Int>
    public let samples: [Float]
}

public enum WordAudioChunkExtractorError: LocalizedError, Equatable {
    case invalidSampleRate(Double)
    case invalidOccurrence(Int)
    case invalidWordIndex(Int)
    case missingTokenTimings
    case wordNotFound(word: String, occurrence: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidSampleRate(let sampleRate):
            "Sample rate must be positive; received \(sampleRate)."
        case .invalidOccurrence(let occurrence):
            "Occurrence must be 1 or greater; received \(occurrence)."
        case .invalidWordIndex(let index):
            "Word index \(index) is outside the timed transcript."
        case .missingTokenTimings:
            "The transcription did not return token timings."
        case .wordNotFound(let word, let occurrence):
            "Could not find occurrence \(occurrence) of '\(word)' in the timed transcript."
        }
    }
}

/// Reconstructs word timings and maps pronunciation matches to the same samples sent to ASR.
///
/// This type is intentionally independent from model loading and transcription. It can be removed without changing
/// the Parakeet inference path, and later reused by pronunciation-embedding experiments.
public enum WordAudioChunkExtractor {
    /// Reconstruct word-level timings from SentencePiece-style token timings.
    public static func words(from tokenTimings: [TokenTiming]) -> [TimedTranscriptWord] {
        var words: [TimedTranscriptWord] = []
        var text = ""
        var startTime: TimeInterval?
        var endTime: TimeInterval = 0
        var confidenceSum: Float = 0
        var confidenceCount = 0

        func appendCurrentWord() {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, let startTime else { return }
            words.append(
                TimedTranscriptWord(
                    text: trimmed,
                    startTime: startTime,
                    endTime: endTime,
                    confidence: confidenceCount > 0 ? confidenceSum / Float(confidenceCount) : 0
                ))
        }

        for timing in tokenTimings {
            let token = timing.token.replacingOccurrences(of: "▁", with: " ")
            guard !token.isEmpty, token != "<blank>", token != "<pad>" else { continue }

            let startsNewWord = token.first?.isWhitespace == true
            if startsNewWord, !text.isEmpty {
                appendCurrentWord()
                text = ""
                startTime = nil
                confidenceSum = 0
                confidenceCount = 0
            }

            if startTime == nil {
                startTime = timing.startTime
            }
            text += startsNewWord ? token.trimmingCharacters(in: .whitespacesAndNewlines) : token
            endTime = max(timing.endTime, timing.startTime)
            confidenceSum += timing.confidence
            confidenceCount += 1
        }

        appendCurrentWord()
        return words
    }

    /// Returns words whose own duration is mostly contained by an acoustic match window.
    ///
    /// Requiring a majority prevents a small encoder-frame spill from claiming an adjacent word while still allowing
    /// one pronunciation match to replace several decoded words.
    public static func substantiallyOverlappingWordIndices(
        in words: [TimedTranscriptWord],
        startTime: TimeInterval,
        endTime: TimeInterval,
        minimumOverlapRatio: Double = PronunciationCustomizationDefaults.minimumWordOverlapRatio
    ) -> [Int] {
        guard endTime > startTime else { return [] }
        let requiredRatio = min(1, max(0, minimumOverlapRatio))
        return words.indices.filter { index in
            let word = words[index]
            let duration = word.endTime - word.startTime
            guard duration > 0 else { return false }
            let overlap = max(0, min(word.endTime, endTime) - max(word.startTime, startTime))
            return overlap / duration > requiredRatio
        }
    }

    /// Select a word occurrence and return its optionally padded audio samples.
    public static func extract(
        word target: String,
        occurrence: Int = 1,
        tokenTimings: [TokenTiming],
        audioSamples: [Float],
        sampleRate: Double = 16_000,
        padding: TimeInterval = 0
    ) throws -> WordAudioChunk {
        guard sampleRate > 0 else {
            throw WordAudioChunkExtractorError.invalidSampleRate(sampleRate)
        }
        guard occurrence > 0 else {
            throw WordAudioChunkExtractorError.invalidOccurrence(occurrence)
        }
        guard !tokenTimings.isEmpty else {
            throw WordAudioChunkExtractorError.missingTokenTimings
        }

        let timedWords = words(from: tokenTimings)
        let normalizedTarget = normalizedWord(target)
        var matchedOccurrence = 0

        for (index, timedWord) in timedWords.enumerated() where normalizedWord(timedWord.text) == normalizedTarget {
            matchedOccurrence += 1
            guard matchedOccurrence == occurrence else { continue }

            return makeChunk(
                word: timedWord,
                wordIndex: index,
                occurrence: occurrence,
                audioSamples: audioSamples,
                sampleRate: sampleRate,
                padding: padding
            )
        }

        throw WordAudioChunkExtractorError.wordNotFound(word: target, occurrence: occurrence)
    }

    /// Select a word by its position in ``words(from:)`` and return its audio samples.
    public static func extract(
        wordAt index: Int,
        tokenTimings: [TokenTiming],
        audioSamples: [Float],
        sampleRate: Double = 16_000,
        padding: TimeInterval = 0
    ) throws -> WordAudioChunk {
        guard sampleRate > 0 else {
            throw WordAudioChunkExtractorError.invalidSampleRate(sampleRate)
        }
        guard !tokenTimings.isEmpty else {
            throw WordAudioChunkExtractorError.missingTokenTimings
        }
        let timedWords = words(from: tokenTimings)
        guard timedWords.indices.contains(index) else {
            throw WordAudioChunkExtractorError.invalidWordIndex(index)
        }
        return makeChunk(
            word: timedWords[index],
            wordIndex: index,
            occurrence: 1,
            audioSamples: audioSamples,
            sampleRate: sampleRate,
            padding: padding
        )
    }

    private static func makeChunk(
        word: TimedTranscriptWord,
        wordIndex: Int,
        occurrence: Int,
        audioSamples: [Float],
        sampleRate: Double,
        padding: TimeInterval
    ) -> WordAudioChunk {
        let audioDuration = TimeInterval(audioSamples.count) / sampleRate
        let extractionStart = max(0, word.startTime - max(0, padding))
        let extractionEnd = min(audioDuration, word.endTime + max(0, padding))
        let startSample = min(
            audioSamples.count,
            max(0, Int((extractionStart * sampleRate).rounded(.toNearestOrAwayFromZero)))
        )
        let endSample = min(
            audioSamples.count,
            max(startSample, Int((extractionEnd * sampleRate).rounded(.toNearestOrAwayFromZero)))
        )
        let range = startSample..<endSample
        return WordAudioChunk(
            word: word,
            wordIndex: wordIndex,
            occurrence: occurrence,
            extractionStartTime: extractionStart,
            extractionEndTime: extractionEnd,
            sampleRange: range,
            samples: Array(audioSamples[range])
        )
    }

    private static func normalizedWord(_ word: String) -> String {
        let scalars = word.lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }
        return String(String.UnicodeScalarView(scalars))
    }
}
