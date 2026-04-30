import Foundation
import XCTest

@testable import FluidAudio

final class GraniteAsrModelsTests: XCTestCase {
    private static let minimalManifest = """
    {
      "model_id": "ibm-granite/granite-speech-4.1-2b-nar",
      "sample_rate": 16000,
      "n_fft": 512,
      "win_length": 400,
      "hop_length": 160,
      "n_mels": 80,
      "feature_frames_per_second": 50,
      "bpe_pooling_window": 4,
      "default_window_seconds": 35,
      "default_overlap_seconds": 5.0,
      "speed_window_seconds": 60,
      "speed_overlap_seconds": 5.0,
      "mel_filters": "granite_mel_filters_80x257_f32.bin",
      "tokenizer": "tokenizer.json",
      "windows": {
        "35s": {
          "seconds": 35,
          "samples": 560000,
          "frames": 1750,
          "bpe_frames": 438,
          "package": "granite_bpe_greedy_35s.mlpackage",
          "inputs": ["input_features", "attention_mask"],
          "outputs": ["bpe_token_ids", "bpe_length"]
        },
        "60s": {
          "seconds": 60,
          "samples": 960000,
          "frames": 3000,
          "bpe_frames": 750,
          "package": "granite_bpe_greedy_60s.mlpackage",
          "inputs": ["input_features", "attention_mask"],
          "outputs": ["bpe_token_ids", "bpe_length"]
        }
      }
    }
    """

    func testGraniteRepoMetadata() {
        XCTAssertEqual(GraniteAsrBundleDownloader.repoPath, "FluidInference/granite-speech-4.1-2b-nar-coreml")
        XCTAssertEqual(GraniteAsrBundleDownloader.folderName, "granite-speech-4.1-2b-nar-coreml")
    }

    func testDefaultCacheDirectoryUsesGraniteFolder() {
        let cacheDirectory = GraniteAsrModels.defaultCacheDirectory()

        XCTAssertEqual(cacheDirectory.lastPathComponent, GraniteAsrBundleDownloader.folderName)
        XCTAssertTrue(cacheDirectory.path.contains("/FluidAudio/Models/"))
    }

    func testModelsExistReturnsFalseForEmptyDirectory() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertFalse(GraniteAsrModels.modelsExist(at: directory))
    }

    func testModelsExistReturnsTrueForMinimalBundleLayout() throws {
        let directory = try makeTemporaryGraniteBundle()
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertTrue(GraniteAsrModels.modelsExist(at: directory))
    }

    func testModelsExistReturnsFalseWhenWindowPackageIsMissing() throws {
        let directory = try makeTemporaryGraniteBundle()
        defer { try? FileManager.default.removeItem(at: directory) }

        try FileManager.default.removeItem(
            at: directory.appendingPathComponent("granite_bpe_greedy_60s.mlpackage")
        )

        XCTAssertFalse(GraniteAsrModels.modelsExist(at: directory))
    }

    private func makeTemporaryGraniteBundle() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try Data(Self.minimalManifest.utf8).write(to: directory.appendingPathComponent("granite_manifest.json"))
        try Data().write(to: directory.appendingPathComponent("granite_mel_filters_80x257_f32.bin"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("tokenizer.json"))
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("granite_bpe_greedy_35s.mlpackage"),
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("granite_bpe_greedy_60s.mlpackage"),
            withIntermediateDirectories: true
        )
        return directory
    }
}
