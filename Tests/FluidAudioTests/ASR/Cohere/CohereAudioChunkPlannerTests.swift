import XCTest
@testable import FluidAudio

@available(macOS 15, iOS 18, *)
final class CohereAudioChunkPlannerTests: XCTestCase {
    func testShortAudioUsesSingleChunk() {
        let samples = Array(repeating: Float(0.1), count: 20 * 16_000)

        let chunks = CohereAudioChunkPlanner.makeChunks(
            audioSamples: samples,
            sampleRate: 16_000,
            maxAudioSamples: 480_000
        )

        XCTAssertEqual(chunks, [.init(startSample: 0, endSample: samples.count)])
    }

    func testLongAudioSplitsAtQuietBoundaryNearChunkEnd() {
        var samples = Array(repeating: Float(0.8), count: 34 * 16_000)
        let quietStart = 27 * 16_000
        let quietEnd = quietStart + 1_600
        for index in quietStart..<quietEnd {
            samples[index] = 0.001
        }

        let chunks = CohereAudioChunkPlanner.makeChunks(
            audioSamples: samples,
            sampleRate: 16_000,
            maxAudioSamples: 480_000
        )

        XCTAssertEqual(chunks.count, 2)
        XCTAssertEqual(chunks[0].endSample, quietStart)
        XCTAssertEqual(chunks[1].startSample, quietStart)
        XCTAssertEqual(chunks[1].endSample, samples.count)
    }

    func testQuietestSplitPointFallsBackToMiddleForShortSearchRange() {
        let samples = Array(repeating: Float(0.3), count: 1_000)

        let split = CohereAudioChunkPlanner.findQuietestSplitPoint(
            samples,
            startSample: 100,
            endSample: 900,
            minEnergyWindowSamples: 1_600
        )

        XCTAssertEqual(split, 500)
    }

    func testJoinChunkTextsAvoidsDuplicateSpacing() {
        let joined = CohereAudioChunkPlanner.joinChunkTexts([
            "Hello world",
            "this is fine",
            ".",
        ])

        XCTAssertEqual(joined, "Hello world this is fine.")
    }

    func testJoinChunkTextsSkipsSpaceForCJKBoundaries() {
        let joined = CohereAudioChunkPlanner.joinChunkTexts([
            "你好",
            "世界",
        ])

        XCTAssertEqual(joined, "你好世界")
    }
}
