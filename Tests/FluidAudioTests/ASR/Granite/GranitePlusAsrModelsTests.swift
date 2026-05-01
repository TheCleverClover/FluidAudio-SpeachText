import Foundation
import XCTest

@testable import FluidAudio

final class GranitePlusAsrModelsTests: XCTestCase {
    private static let minimalManifest = """
        {
          "model_id": "ibm-granite/granite-speech-4.1-2b-plus",
          "sample_rate": 16000,
          "n_fft": 512,
          "win_length": 400,
          "hop_length": 160,
          "n_mels": 80,
          "feature_frames_per_second": 50,
          "mel_filters": "granite_mel_filters_80x257_f32.bin",
          "tokenizer": "tokenizer.json",
          "token_embedding_package": "granite_plus_token_embed_q1024_fp16.mlpackage",
          "language_model_package": "granite_plus_lm_flexible_stateful_m1024_q1024_fp16.mlpackage",
          "max_sequence_length": 1024,
          "max_query_length": 1024,
          "audio_token_id": 100352,
          "eos_token_id": 100257,
          "default_window_seconds": 15,
          "windows": {
            "15s": {
              "seconds": 15,
              "samples": 240000,
              "frames": 750,
              "audio_tokens": 150,
              "package": "granite_plus_audio_15s.mlpackage",
              "inputs": ["input_features"],
              "outputs": ["audio_embeds"]
            }
          }
        }
        """

    func testPromptBuilderRepeatsAudioTokens() {
        let prompt = GranitePlusPromptBuilder.prompt(task: .timestamp, audioTokenCount: 3)

        XCTAssertEqual(prompt.components(separatedBy: "<|audio|>").count - 1, 3)
        XCTAssertTrue(prompt.contains("<|start_of_role|>system<|end_of_role|>"))
        XCTAssertTrue(prompt.contains("<|start_of_role|>assistant<|end_of_role|>"))
        XCTAssertTrue(prompt.contains("Timestamps: Transcribe the speech."))
    }

    func testChunkMergeRemovesFuzzyOverlap() {
        if #available(macOS 15, iOS 18, *) {
            let left = "when the buzzer sounded he pulled his foil from his second startled grasp"
            let right = "when the buzzer sounded he pulled his foil from his second's startled grasp and ran forward"

            let merged = GranitePlusChunkMerge.mergeTranscriptChunks([left, right])

            XCTAssertEqual(
                merged,
                "when the buzzer sounded he pulled his foil from his second startled grasp and ran forward"
            )
        }
    }

    func testModelsExistReturnsTrueForMinimalPlusBundle() throws {
        if #available(macOS 15, iOS 18, *) {
            let directory = try makeTemporaryGranitePlusBundle()
            defer { try? FileManager.default.removeItem(at: directory) }

            XCTAssertTrue(GranitePlusAsrModels.modelsExist(at: directory))
        }
    }

    func testModelsExistReturnsFalseWhenPlusLmPackageIsMissing() throws {
        if #available(macOS 15, iOS 18, *) {
            let directory = try makeTemporaryGranitePlusBundle()
            defer { try? FileManager.default.removeItem(at: directory) }

            try FileManager.default.removeItem(
                at: directory.appendingPathComponent("granite_plus_lm_flexible_stateful_m1024_q1024_fp16.mlpackage")
            )

            XCTAssertFalse(GranitePlusAsrModels.modelsExist(at: directory))
        }
    }

    private func makeTemporaryGranitePlusBundle() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try Data(Self.minimalManifest.utf8).write(to: directory.appendingPathComponent("granite_plus_manifest.json"))
        try Data().write(to: directory.appendingPathComponent("granite_mel_filters_80x257_f32.bin"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("tokenizer.json"))
        for package in [
            "granite_plus_audio_15s.mlpackage",
            "granite_plus_token_embed_q1024_fp16.mlpackage",
            "granite_plus_lm_flexible_stateful_m1024_q1024_fp16.mlpackage",
        ] {
            try FileManager.default.createDirectory(
                at: directory.appendingPathComponent(package),
                withIntermediateDirectories: true
            )
        }
        return directory
    }
}
