import Foundation
import Testing

@testable import FluidAudio

@Suite("Parakeet pronunciation word alignment")
struct WordAudioChunkExtractorTests {
    @Test("SentencePiece tokens merge into timed words")
    func mergesTokensIntoWords() {
        let timings = [
            timing(" Hel", start: 0.08, end: 0.16, confidence: 0.8),
            timing("lo", start: 0.16, end: 0.32, confidence: 1.0),
            timing("▁Bharat", start: 0.40, end: 0.64, confidence: 0.9),
            timing("waj", start: 0.64, end: 0.88, confidence: 0.7),
        ]

        let words = WordAudioChunkExtractor.words(from: timings)

        #expect(words.count == 2)
        #expect(words[0].text == "Hello")
        #expect(words[0].startTime == 0.08)
        #expect(words[0].endTime == 0.32)
        #expect(abs(words[0].confidence - 0.9) < 0.0001)
        #expect(words[1].text == "Bharatwaj")
        #expect(words[1].startTime == 0.40)
        #expect(words[1].endTime == 0.88)
        #expect(abs(words[1].confidence - 0.8) < 0.0001)
    }

    @Test("Extraction selects an occurrence and clamps padded sample bounds")
    func extractsSelectedOccurrence() throws {
        let timings = [
            timing(" Test", start: 0.0, end: 0.2),
            timing(" test", start: 0.7, end: 0.9),
        ]
        let samples = (0..<1_000).map(Float.init)

        let chunk = try WordAudioChunkExtractor.extract(
            word: "test!",
            occurrence: 2,
            tokenTimings: timings,
            audioSamples: samples,
            sampleRate: 1_000,
            padding: 0.2
        )

        #expect(chunk.wordIndex == 1)
        #expect(abs(chunk.extractionStartTime - 0.5) < 0.0001)
        #expect(abs(chunk.extractionEndTime - 1.0) < 0.0001)
        #expect(chunk.sampleRange == 500..<1_000)
        #expect(chunk.samples.count == 500)
        #expect(chunk.samples.first == 500)
    }

    @Test("Missing target reports a useful error")
    func reportsMissingTarget() {
        #expect(throws: WordAudioChunkExtractorError.wordNotFound(word: "Bharatwaj", occurrence: 1)) {
            try WordAudioChunkExtractor.extract(
                word: "Bharatwaj",
                tokenTimings: [timing(" hello", start: 0, end: 0.2)],
                audioSamples: Array(repeating: 0, count: 16_000)
            )
        }
    }

    @Test("Acoustic match excludes a lightly overlapping previous word")
    func excludesBoundarySpill() {
        let words = [
            TimedTranscriptWord(text: "and", startTime: 2.16, endTime: 2.40, confidence: 1),
            TimedTranscriptWord(text: "Fluid", startTime: 2.32, endTime: 2.64, confidence: 1),
            TimedTranscriptWord(text: "Write", startTime: 2.64, endTime: 2.88, confidence: 1),
        ]

        let indices = WordAudioChunkExtractor.substantiallyOverlappingWordIndices(
            in: words,
            startTime: 2.32,
            endTime: 2.88
        )

        #expect(indices == [1, 2])
    }

    @Test("Acoustic match retains multiple decoded words")
    func retainsSplitPronunciation() {
        let words = [
            TimedTranscriptWord(text: "Barat", startTime: 0.40, endTime: 0.72, confidence: 1),
            TimedTranscriptWord(text: "watch", startTime: 0.72, endTime: 1.04, confidence: 1),
        ]

        let indices = WordAudioChunkExtractor.substantiallyOverlappingWordIndices(
            in: words,
            startTime: 0.48,
            endTime: 0.96
        )

        #expect(indices == [0, 1])
    }

    private func timing(
        _ token: String,
        start: TimeInterval,
        end: TimeInterval,
        confidence: Float = 1
    ) -> TokenTiming {
        TokenTiming(token: token, tokenId: 1, startTime: start, endTime: end, confidence: confidence)
    }
}
