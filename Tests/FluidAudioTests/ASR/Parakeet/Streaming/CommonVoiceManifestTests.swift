#if os(macOS)
import Foundation
import XCTest

@testable import FluidAudio

final class CommonVoiceManifestTests: XCTestCase {
    func testLoadsLocaleSplitAndResolvesClipsDirectory() throws {
        let root = try makeTempDirectory()
        let localeDir = root.appendingPathComponent("pt")
        let clipsDir = localeDir.appendingPathComponent("clips")
        try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)

        let audioURL = clipsDir.appendingPathComponent("sample.mp3")
        FileManager.default.createFile(atPath: audioURL.path, contents: Data())

        let manifest = """
            client_id\tpath\tsentence\tlocale
            a\tsample.mp3\tola mundo\tpt
            """
        try manifest.write(to: localeDir.appendingPathComponent("test.tsv"), atomically: true, encoding: .utf8)

        let samples = try CommonVoiceManifest.loadSamples(
            datasetDirectory: root,
            split: "test",
            language: "pt"
        )

        XCTAssertEqual(samples.count, 1)
        XCTAssertEqual(samples[0].sampleId, "sample")
        XCTAssertEqual(samples[0].audioPath, audioURL)
        XCTAssertEqual(samples[0].transcript, "ola mundo")
        XCTAssertEqual(samples[0].metadata["locale"], "pt")
    }

    func testVariantFilterUsesVariantLocaleOrAccentColumns() throws {
        let root = try makeTempDirectory()
        let localeDir = root.appendingPathComponent("pt")
        let clipsDir = localeDir.appendingPathComponent("clips")
        try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: clipsDir.appendingPathComponent("br.mp3").path, contents: Data())
        FileManager.default.createFile(atPath: clipsDir.appendingPathComponent("pt.mp3").path, contents: Data())

        let manifest = """
            path\tsentence\tvariant\taccents
            br.mp3\tfala brasileira\tpt-BR\tBrazilian
            pt.mp3\tfala portuguesa\tpt-PT\tPortugal
            """
        try manifest.write(to: localeDir.appendingPathComponent("validated.tsv"), atomically: true, encoding: .utf8)

        let samples = try CommonVoiceManifest.loadSamples(
            datasetDirectory: root,
            split: "validated",
            language: "pt",
            variant: "pt-BR"
        )

        XCTAssertEqual(samples.map(\.relativePath), ["br.mp3"])
        XCTAssertEqual(samples[0].transcript, "fala brasileira")
    }

    func testThrowsWhenNoManifestExists() throws {
        let root = try makeTempDirectory()

        XCTAssertThrowsError(
            try CommonVoiceManifest.loadSamples(datasetDirectory: root, split: "test", language: "pt")
        )
    }

    private func makeTempDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: url)
        }
        return url
    }
}
#endif
