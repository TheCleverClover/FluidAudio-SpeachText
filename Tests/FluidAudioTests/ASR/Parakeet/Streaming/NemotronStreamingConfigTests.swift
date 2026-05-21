import Foundation
import XCTest

@testable import FluidAudio

final class NemotronStreamingConfigTests: XCTestCase {

    // MARK: - P0: Default Initialization

    func testDefaultInitialization() {
        let config = NemotronStreamingConfig()

        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.melFeatures, 128)
        XCTAssertEqual(config.chunkMelFrames, 112)
        XCTAssertEqual(config.chunkMs, 1120)
        XCTAssertEqual(config.preEncodeCache, 9)
        XCTAssertEqual(config.totalMelFrames, 121)
        XCTAssertEqual(config.vocabSize, 1024)
        XCTAssertEqual(config.blankIdx, 1024)
        XCTAssertEqual(config.encoderDim, 1024)
        XCTAssertEqual(config.decoderHidden, 640)
        XCTAssertEqual(config.decoderLayers, 2)
        XCTAssertEqual(config.cacheChannelShape, [1, 24, 70, 1024])
        XCTAssertEqual(config.cacheTimeShape, [1, 24, 1024, 8])
    }

    func testChunkSamplesComputation() {
        let config = NemotronStreamingConfig()
        // chunkSamples = chunkMelFrames * 160
        XCTAssertEqual(config.chunkSamples, 112 * 160)
        XCTAssertEqual(config.chunkSamples, 17920)
    }

    // MARK: - P0: JSON Loading - Valid Cases

    func testLoadValidMetadataJson1120ms() throws {
        let metadata: [String: Any] = [
            "sample_rate": 16000,
            "mel_features": 128,
            "chunk_mel_frames": 112,
            "chunk_ms": 1120,
            "pre_encode_cache": 9,
            "total_mel_frames": 121,
            "vocab_size": 1024,
            "blank_idx": 1024,
            "encoder_dim": 1024,
            "decoder_hidden": 640,
            "decoder_layers": 2,
            "cache_channel_shape": [1, 24, 70, 1024],
            "cache_time_shape": [1, 24, 1024, 8],
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.melFeatures, 128)
        XCTAssertEqual(config.chunkMelFrames, 112)
        XCTAssertEqual(config.chunkMs, 1120)
        XCTAssertEqual(config.preEncodeCache, 9)
        XCTAssertEqual(config.totalMelFrames, 121)
        XCTAssertEqual(config.vocabSize, 1024)
        XCTAssertEqual(config.blankIdx, 1024)
        XCTAssertEqual(config.encoderDim, 1024)
        XCTAssertEqual(config.decoderHidden, 640)
        XCTAssertEqual(config.decoderLayers, 2)
        XCTAssertEqual(config.cacheChannelShape, [1, 24, 70, 1024])
        XCTAssertEqual(config.cacheTimeShape, [1, 24, 1024, 8])
    }

    func testLoadValidMetadataJson80ms() throws {
        let metadata: [String: Any] = [
            "sample_rate": 16000,
            "mel_features": 128,
            "chunk_mel_frames": 8,
            "chunk_ms": 80,
            "pre_encode_cache": 4,
            "total_mel_frames": 12,
            "vocab_size": 1024,
            "blank_idx": 1024,
            "encoder_dim": 1024,
            "decoder_hidden": 640,
            "decoder_layers": 2,
            "cache_channel_shape": [1, 24, 120, 1024],  // Larger cache for tiny chunks
            "cache_time_shape": [1, 24, 1024, 10],
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        XCTAssertEqual(config.chunkMelFrames, 8)
        XCTAssertEqual(config.chunkMs, 80)
        XCTAssertEqual(config.preEncodeCache, 4)
        XCTAssertEqual(config.totalMelFrames, 12)
        XCTAssertEqual(config.chunkSamples, 8 * 160)  // 1280 samples
        XCTAssertEqual(config.cacheChannelShape, [1, 24, 120, 1024])
    }

    func testLoadValidMetadataJson560ms() throws {
        let metadata: [String: Any] = [
            "chunk_mel_frames": 56,
            "chunk_ms": 560,
            "pre_encode_cache": 7,
            "total_mel_frames": 63,
            "cache_channel_shape": [1, 24, 85, 1024],
            "cache_time_shape": [1, 24, 1024, 9],
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        XCTAssertEqual(config.chunkMelFrames, 56)
        XCTAssertEqual(config.chunkMs, 560)
        XCTAssertEqual(config.preEncodeCache, 7)
        XCTAssertEqual(config.totalMelFrames, 63)
        XCTAssertEqual(config.chunkSamples, 56 * 160)  // 8960 samples
    }

    func testLoadSingleEncoderMetadataJson() throws {
        let metadata: [String: Any] = [
            "model_layout": "single_encoder",
            "sample_rate": 16000,
            "vocab_size": 13087,
            "blank_idx": 13087,
            "decoder_hidden": 640,
            "decoder_layers": 2,
            "max_audio_samples": 240000,
            "coreml": [
                "components": [
                    "preprocessor": "preprocessor.mlpackage",
                    "encoder": "encoder.mlpackage",
                    "decoder": "decoder.mlpackage",
                    "joint": "joint.mlpackage",
                ]
            ],
            "shapes": [
                "audio_signal": [1, 240000],
                "processed_signal_step": [1, 128, 17],
                "cache_last_channel": [1, 24, 56, 1024],
                "cache_last_time": [1, 24, 1024, 8],
                "encoded_step": [1, 1024, 1],
            ],
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        XCTAssertEqual(config.modelLayout, .singleEncoder)
        XCTAssertEqual(config.melFeatures, 128)
        XCTAssertEqual(config.preEncodeCache, 9)
        XCTAssertEqual(config.totalMelFrames, 17)
        XCTAssertEqual(config.chunkMelFrames, 8)
        XCTAssertEqual(config.chunkMs, 80)
        XCTAssertEqual(config.chunkSamples, 1280)
        XCTAssertEqual(config.vocabSize, 13087)
        XCTAssertEqual(config.blankIdx, 13087)
        XCTAssertEqual(config.maxAudioSamples, 240000)
        XCTAssertEqual(config.cacheChannelShape, [1, 24, 56, 1024])
        XCTAssertEqual(config.cacheTimeShape, [1, 24, 1024, 8])
    }

    func testSingleEncoderFlexiblePreprocessorDoesNotForceMaxAudioPadding() throws {
        let metadata: [String: Any] = [
            "max_audio_samples": 240000,
            "coreml": [
                "preprocessor_audio_flexible": true,
                "components": [
                    "encoder": "encoder.mlpackage"
                ],
            ],
            "shapes": [
                "audio_signal": [1, 240000],
                "processed_signal_step": [1, 128, 17],
            ],
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        XCTAssertEqual(config.modelLayout, .singleEncoder)
        XCTAssertNil(config.maxAudioSamples)
        XCTAssertEqual(config.chunkSamples, 1280)
    }

    func testSingleEncoderLargerStreamingWindowUsesMetadataShape() throws {
        let metadata: [String: Any] = [
            "coreml": [
                "preprocessor_audio_flexible": true,
                "components": [
                    "encoder": "encoder.mlpackage"
                ],
            ],
            "shapes": [
                "processed_signal_step": [1, 128, 137],
                "encoded_step": [1, 1024, 16],
            ],
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        XCTAssertEqual(config.modelLayout, .singleEncoder)
        XCTAssertEqual(config.totalMelFrames, 137)
        XCTAssertEqual(config.chunkMelFrames, 128)
        XCTAssertEqual(config.chunkMs, 1280)
        XCTAssertEqual(config.chunkSamples, 20480)
        XCTAssertNil(config.maxAudioSamples)
    }

    func testRuntimePromptMetadataParsing() throws {
        let metadata: [String: Any] = [
            "runtime_prompt": true,
            "target_lang": "pt-BR",
            "num_prompts": 128,
            "prompt_dictionary": [
                "en-US": 0,
                "pt-BR": 12,
                "pt": 13,
                "auto": 101,
            ],
            "coreml": [
                "preprocessor_audio_flexible": true,
                "components": [
                    "encoder": "encoder.mlpackage"
                ],
            ],
            "shapes": [
                "processed_signal_step": [1, 128, 137],
                "encoded_step": [1, 1024, 16],
            ],
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        XCTAssertTrue(config.runtimePrompt)
        XCTAssertEqual(config.targetLang, "pt-BR")
        XCTAssertEqual(config.numPrompts, 128)
        XCTAssertEqual(config.promptDictionary["pt-BR"], 12)
        XCTAssertEqual(config.promptDictionary["pt"], 13)
        XCTAssertEqual(config.promptDictionary["auto"], 101)
    }

    // MARK: - P0: JSON Loading - Fallback Defaults

    func testLoadPartialJsonUsesDefaults() throws {
        let metadata: [String: Any] = [
            "chunk_mel_frames": 56  // Only this field present
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        // Custom value
        XCTAssertEqual(config.chunkMelFrames, 56)

        // Defaults
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.melFeatures, 128)
        XCTAssertEqual(config.vocabSize, 1024)
        XCTAssertEqual(config.blankIdx, 1024)
    }

    func testLoadEmptyJsonUsesAllDefaults() throws {
        let metadata: [String: Any] = [:]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        // All defaults (same as default init)
        XCTAssertEqual(config.chunkMelFrames, 112)
        XCTAssertEqual(config.chunkMs, 1120)
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.vocabSize, 1024)
    }

    // MARK: - P0: JSON Loading - Error Cases

    func testLoadInvalidJsonFormatThrows() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let file = tempDir.appendingPathComponent("invalid_\(UUID().uuidString).json")

        try "not json at all".write(to: file, atomically: true, encoding: .utf8)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: file)
        }

        XCTAssertThrowsError(try NemotronStreamingConfig(from: file)) { error in
            // Should throw JSON parsing error
            XCTAssertTrue(error is NSError || error is DecodingError)
        }
    }

    func testLoadJsonArrayInsteadOfDictionaryThrows() throws {
        let jsonArray = try JSONSerialization.data(withJSONObject: [1, 2, 3], options: [])
        let file = try createTempFile(jsonArray)

        XCTAssertThrowsError(try NemotronStreamingConfig(from: file)) { error in
            guard case ASRError.processingFailed(let message) = error else {
                XCTFail("Expected ASRError.processingFailed, got \(error)")
                return
            }
            XCTAssertEqual(message, "Invalid metadata.json format")
        }
    }

    func testLoadNonExistentFileThrows() {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(
            "nonexistent_\(UUID().uuidString).json")

        XCTAssertThrowsError(try NemotronStreamingConfig(from: file))
    }

    // MARK: - P0: Type Coercion

    func testLoadJsonWithWrongTypesUsesDefaults() throws {
        let metadata: [String: Any] = [
            "chunk_mel_frames": "not a number",  // Wrong type
            "sample_rate": 16000,  // Correct type
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        // Wrong type → uses default
        XCTAssertEqual(config.chunkMelFrames, 112)  // Default value

        // Correct type → uses provided
        XCTAssertEqual(config.sampleRate, 16000)
    }

    func testLoadJsonWithArrayTypesCorrectly() throws {
        let metadata: [String: Any] = [
            "cache_channel_shape": [1, 24, 100, 2048],
            "cache_time_shape": [1, 24, 2048, 16],
        ]

        let file = try createTempJsonFile(metadata)
        let config = try NemotronStreamingConfig(from: file)

        XCTAssertEqual(config.cacheChannelShape, [1, 24, 100, 2048])
        XCTAssertEqual(config.cacheTimeShape, [1, 24, 2048, 16])
    }

    // MARK: - Helpers

    private func createTempJsonFile(_ dict: [String: Any]) throws -> URL {
        let data = try JSONSerialization.data(withJSONObject: dict, options: [])
        return try createTempFile(data)
    }

    private func createTempFile(_ data: Data) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let file = tempDir.appendingPathComponent("test_metadata_\(UUID().uuidString).json")
        try data.write(to: file)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: file)
        }
        return file
    }
}
