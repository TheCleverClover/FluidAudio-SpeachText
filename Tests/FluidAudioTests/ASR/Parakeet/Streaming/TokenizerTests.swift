import Foundation
import XCTest

@testable import FluidAudio

final class TokenizerTests: XCTestCase {

    // MARK: - Helpers

    private func createTempVocabFile(_ vocab: [String: String]) throws -> URL {
        let data = try JSONSerialization.data(withJSONObject: vocab, options: [])
        let tempDir = FileManager.default.temporaryDirectory
        let file = tempDir.appendingPathComponent("test_vocab_\(UUID().uuidString).json")
        try data.write(to: file)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: file)
        }
        return file
    }

    private func createTempSentencePieceModel(_ pieces: [String]) throws -> URL {
        var data = Data()
        for piece in pieces {
            let field = try sentencePieceField(piece)
            data.append(0x0A)
            data.append(contentsOf: varint(field.count))
            data.append(field)
        }

        let tempDir = FileManager.default.temporaryDirectory
        let file = tempDir.appendingPathComponent("test_tokenizer_\(UUID().uuidString).model")
        try data.write(to: file)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: file)
        }
        return file
    }

    private func sentencePieceField(_ piece: String) throws -> Data {
        let bytes = Array(piece.utf8)
        var data = Data([0x0A])
        data.append(contentsOf: varint(bytes.count))
        data.append(contentsOf: bytes)
        return data
    }

    private func varint(_ value: Int) -> [UInt8] {
        var value = value
        var bytes: [UInt8] = []
        repeat {
            var byte = UInt8(value & 0x7F)
            value >>= 7
            if value != 0 {
                byte |= 0x80
            }
            bytes.append(byte)
        } while value != 0
        return bytes
    }

    // MARK: - Decode Known Token IDs

    func testDecodeKnownTokenIds() throws {
        let vocab: [String: String] = [
            "0": "\u{2581}Hello",
            "1": "\u{2581}world",
        ]
        let file = try createTempVocabFile(vocab)
        let tokenizer = try Tokenizer(vocabPath: file)

        let result = tokenizer.decode(ids: [0, 1])
        XCTAssertEqual(result, "Hello world")
    }

    func testDecodeUnknownTokenIdIsSkipped() throws {
        let vocab: [String: String] = [
            "0": "\u{2581}Hello"
        ]
        let file = try createTempVocabFile(vocab)
        let tokenizer = try Tokenizer(vocabPath: file)

        let result = tokenizer.decode(ids: [0, 999])
        XCTAssertEqual(result, "Hello")
    }

    func testDecodeEmptyIdsReturnsEmpty() throws {
        let vocab: [String: String] = [
            "0": "\u{2581}Hello"
        ]
        let file = try createTempVocabFile(vocab)
        let tokenizer = try Tokenizer(vocabPath: file)

        let result = tokenizer.decode(ids: [])
        XCTAssertEqual(result, "")
    }

    func testSentencePieceBoundaryReplacement() throws {
        let vocab: [String: String] = [
            "0": "\u{2581}The",
            "1": "\u{2581}quick",
            "2": "\u{2581}brown",
        ]
        let file = try createTempVocabFile(vocab)
        let tokenizer = try Tokenizer(vocabPath: file)

        let result = tokenizer.decode(ids: [0, 1, 2])
        XCTAssertEqual(result, "The quick brown")
    }

    func testDecodeCanSkipAngleBracketSpecialTokens() throws {
        let vocab: [String: String] = [
            "0": "<en-US>",
            "1": "\u{2581}Hello",
            "2": "<pad>",
            "3": "\u{2581}world",
        ]
        let file = try createTempVocabFile(vocab)
        let tokenizer = try Tokenizer(vocabPath: file)

        XCTAssertEqual(tokenizer.decode(ids: [0, 1, 2, 3]), "<en-US> Hello<pad> world")
        XCTAssertEqual(tokenizer.decode(ids: [0, 1, 2, 3], skipSpecialTokens: true), "Hello world")
    }

    func testDecodeSentencePieceModel() throws {
        let file = try createTempSentencePieceModel([
            "<unk>",
            "<en-US>",
            "\u{2581}Hello",
            "\u{2581}world",
        ])
        let tokenizer = try Tokenizer(sentencePiecePath: file)

        XCTAssertEqual(tokenizer.decode(ids: [1, 2, 3]), "<en-US> Hello world")
        XCTAssertEqual(tokenizer.decode(ids: [1, 2, 3], skipSpecialTokens: true), "Hello world")
    }

    func testInvalidJsonThrows() {
        let tempDir = FileManager.default.temporaryDirectory
        let file = tempDir.appendingPathComponent("bad_vocab_\(UUID().uuidString).json")

        do {
            try "not json at all".write(to: file, atomically: true, encoding: .utf8)
            addTeardownBlock {
                try? FileManager.default.removeItem(at: file)
            }
        } catch {
            XCTFail("Failed to write temp file: \(error)")
            return
        }

        XCTAssertThrowsError(try Tokenizer(vocabPath: file))
    }

    func testNonExistentFileThrows() {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(
            "nonexistent_\(UUID().uuidString).json")
        XCTAssertThrowsError(try Tokenizer(vocabPath: file))
    }
}
