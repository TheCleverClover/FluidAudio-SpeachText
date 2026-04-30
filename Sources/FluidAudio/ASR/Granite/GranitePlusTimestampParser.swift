import Foundation

public struct GranitePlusTimestampToken: Equatable, Sendable {
    public let text: String
    public let endSeconds: Double

    public init(text: String, endSeconds: Double) {
        self.text = text
        self.endSeconds = endSeconds
    }
}

public enum GranitePlusTimestampParser {
    public static func parse(
        _ text: String,
        chunkOffsetSeconds: Double = 0
    ) -> [GranitePlusTimestampToken] {
        var cursor = text.startIndex
        var rolloverSeconds = 0.0
        var lastEndSeconds = 0.0
        var tokens: [GranitePlusTimestampToken] = []

        while let openRange = text[cursor...].range(of: "[T:") {
            guard let closeIndex = text[openRange.upperBound...].firstIndex(of: "]") else {
                break
            }

            let wordText = cleanTokenText(String(text[cursor..<openRange.lowerBound]))
            let tagText = text[openRange.upperBound..<closeIndex]
            if let centiseconds = Int(tagText) {
                var localEndSeconds = Double(centiseconds) / 100.0
                while localEndSeconds + rolloverSeconds < lastEndSeconds {
                    rolloverSeconds += 10.0
                }
                localEndSeconds += rolloverSeconds
                lastEndSeconds = localEndSeconds

                if wordText.isEmpty == false {
                    tokens.append(
                        GranitePlusTimestampToken(
                            text: wordText,
                            endSeconds: chunkOffsetSeconds + localEndSeconds
                        )
                    )
                }
            }

            cursor = text.index(after: closeIndex)
        }

        return tokens
    }

    public static func plainText(from text: String) -> String {
        var cursor = text.startIndex
        var pieces: [String] = []

        while let openRange = text[cursor...].range(of: "[T:") {
            pieces.append(String(text[cursor..<openRange.lowerBound]))
            guard let closeIndex = text[openRange.upperBound...].firstIndex(of: "]") else {
                cursor = openRange.upperBound
                break
            }
            cursor = text.index(after: closeIndex)
        }
        if cursor < text.endIndex {
            pieces.append(String(text[cursor..<text.endIndex]))
        }

        return cleanTokenText(pieces.joined(separator: " "))
    }

    public static func merge(
        _ chunks: [[GranitePlusTimestampToken]],
        dedupeWindowSeconds: Double = 0.35
    ) -> [GranitePlusTimestampToken] {
        var merged: [GranitePlusTimestampToken] = []

        for chunk in chunks {
            for token in chunk {
                let normalized = normalize(token.text)
                let isDuplicate = merged.suffix(8).contains { previous in
                    abs(previous.endSeconds - token.endSeconds) <= dedupeWindowSeconds
                        && normalize(previous.text) == normalized
                }
                if isDuplicate == false {
                    merged.append(token)
                }
            }
        }

        return merged.sorted { lhs, rhs in
            lhs.endSeconds < rhs.endSeconds
        }
    }

    private static func cleanTokenText(_ text: String) -> String {
        text
            .replacingOccurrences(of: "_", with: " ")
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
    }

    private static func normalize(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil)
    }
}
