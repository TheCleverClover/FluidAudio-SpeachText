import CoreML
import XCTest

@testable import FluidAudio

final class NemotronCacheAwareMelChunkerTests: XCTestCase {
    func testSplitEncoderMetadataParsing() throws {
        let metadata: [String: Any] = [
            "vocab_size": 13087,
            "blank_idx": 13087,
            "runtime_prompt": true,
            "pad_and_drop_preencoded": false,
            "coreml": [
                "components": [
                    "encoder_init": "encoder_init.mlpackage",
                    "encoder_step": "encoder_step.mlpackage",
                    "preprocessor": "preprocessor.mlpackage",
                ],
            ],
            "shapes": [
                "processed_signal_init": [1, 128, 217],
                "processed_signal_step": [1, 128, 233],
                "cache_last_channel": [24, 1, 56, 1024],
                "cache_last_time": [24, 1, 1024, 8],
            ],
            "streaming": [
                "shift_size": [105, 112],
                "chunk_size": [217, 224],
                "pre_encode_cache_size": [0, 9],
            ],
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        XCTAssertEqual(config.modelLayout, .splitEncoder)
        XCTAssertEqual(config.initTotalMelFrames, 217)
        XCTAssertEqual(config.stepTotalMelFrames, 233)
        XCTAssertEqual(config.initPreEncodeCache, 0)
        XCTAssertEqual(config.preEncodeCache, 9)
        XCTAssertEqual(config.shiftMelFrames, 112)
        XCTAssertEqual(config.streamingShiftSizes, [105, 112])
        XCTAssertFalse(config.padAndDropPreencoded)
    }

    func testMelChunkStartsFollowNeMoShiftPattern() throws {
        let config = try makeSplitConfig()
        let mel = try MLMultiArray(shape: [1, 128, 400], dataType: .float32)
        let chunks = try NemotronCacheAwareMelChunker.makeChunks(
            from: mel,
            validFrames: 350,
            config: config
        )

        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        XCTAssertTrue(chunks[0].isFirstChunk)
        XCTAssertEqual(chunks[0].targetMelFrames, 217)
        XCTAssertEqual(chunks[1].targetMelFrames, 233)
        XCTAssertEqual(chunks[0].validMelFrames, 217)
        XCTAssertGreaterThan(chunks[1].validMelFrames, 0)
    }

    private func makeSplitConfig() throws -> NemotronStreamingConfig {
        let metadata: [String: Any] = [
            "coreml": [
                "components": [
                    "encoder_init": "encoder_init.mlpackage",
                    "encoder_step": "encoder_step.mlpackage",
                ],
            ],
            "shapes": [
                "processed_signal_init": [1, 128, 217],
                "processed_signal_step": [1, 128, 233],
            ],
            "streaming": [
                "shift_size": [105, 112],
            ],
        ]
        return try NemotronStreamingConfig(from: createTempJsonFile(metadata))
    }

    private func createTempJsonFile(_ metadata: [String: Any]) throws -> URL {
        let data = try JSONSerialization.data(withJSONObject: metadata)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("nemotron-metadata-\(UUID().uuidString).json")
        try data.write(to: url)
        return url
    }
}
