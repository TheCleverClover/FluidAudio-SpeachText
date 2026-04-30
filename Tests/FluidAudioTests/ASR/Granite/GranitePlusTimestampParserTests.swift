import Foundation
import Testing

@testable import FluidAudio

@Suite("Granite Plus timestamp parser")
struct GranitePlusTimestampParserTests {
    @Test("unwraps 10 second rollover")
    func unwrapsRollover() {
        let tokens = GranitePlusTimestampParser.parse("hello [T:980] world [T:12] again [T:44]")

        #expect(tokens.map(\.text) == ["hello", "world", "again"])
        #expect(tokens.map { round($0.endSeconds * 100) / 100 } == [9.8, 10.12, 10.44])
    }

    @Test("strips timestamp tags and silence markers")
    func stripsTags() {
        let text = GranitePlusTimestampParser.plainText(from: "hello [T:10] _ [T:20] world [T:30]")

        #expect(text == "hello world")
    }

    @Test("deduplicates overlapped timestamp tokens")
    func deduplicatesOverlap() {
        let left = GranitePlusTimestampParser.parse("hello [T:10] world [T:20]")
        let right = GranitePlusTimestampParser.parse("world [T:21] again [T:40]")
        let merged = GranitePlusTimestampParser.merge([left, right])

        #expect(merged.map(\.text) == ["hello", "world", "again"])
    }
}
