import Foundation
import XCTest

@testable import FluidAudio

final class GranitePlusRuntimeSmokeTests: XCTestCase {
    func testGranitePlusTranscribesRealSampleWhenConfigured() async throws {
        guard #available(macOS 15, iOS 18, *) else {
            throw XCTSkip("Granite Plus CoreML stateful models require macOS 15/iOS 18")
        }
        guard
            let bundlePath = ProcessInfo.processInfo.environment["FLUIDAUDIO_GRANITE_PLUS_BUNDLE"],
            let audioPath = ProcessInfo.processInfo.environment["FLUIDAUDIO_GRANITE_PLUS_AUDIO"]
        else {
            throw XCTSkip("Set FLUIDAUDIO_GRANITE_PLUS_BUNDLE and FLUIDAUDIO_GRANITE_PLUS_AUDIO for real runtime smoke")
        }

        let manager = GranitePlusAsrManager()
        try await manager.loadModels(from: URL(fileURLWithPath: bundlePath), computeUnits: .cpuAndGPU)

        let result = try await manager.transcribeDetailed(
            audioFileAt: URL(fileURLWithPath: audioPath),
            task: .asr,
            maxNewTokens: 96
        )

        XCTAssertTrue(result.stoppedOnEOS)
        XCTAssertEqual(result.chunkCount, 1)
        XCTAssertTrue(result.text.lowercased().contains("timothy lazily stretched"))
        XCTAssertTrue(result.text.lowercased().contains("clean hearth"))
    }

    func testGranitePlusTranscribesLongAudioWhenConfigured() async throws {
        guard #available(macOS 15, iOS 18, *) else {
            throw XCTSkip("Granite Plus CoreML stateful models require macOS 15/iOS 18")
        }
        guard
            let bundlePath = ProcessInfo.processInfo.environment["FLUIDAUDIO_GRANITE_PLUS_BUNDLE"],
            let audioPath = ProcessInfo.processInfo.environment["FLUIDAUDIO_GRANITE_PLUS_LONG_AUDIO"]
        else {
            throw XCTSkip(
                "Set FLUIDAUDIO_GRANITE_PLUS_BUNDLE and FLUIDAUDIO_GRANITE_PLUS_LONG_AUDIO for long runtime smoke")
        }

        let manager = GranitePlusAsrManager()
        try await manager.loadModels(from: URL(fileURLWithPath: bundlePath), computeUnits: .cpuAndGPU)

        let result = try await manager.transcribeDetailed(
            audioFileAt: URL(fileURLWithPath: audioPath),
            task: .asr,
            maxNewTokens: 256
        )

        XCTAssertGreaterThan(result.chunkCount, 1)
        XCTAssertTrue(result.stoppedOnEOS)
        XCTAssertTrue(result.text.lowercased().contains("mister quilter"))
        XCTAssertTrue(result.text.lowercased().contains("islington cabman"))
    }
}
