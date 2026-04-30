import Foundation
import XCTest

@testable import FluidAudio

final class GraniteAsrModelsTests: XCTestCase {
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
}
