import Foundation

@available(macOS 15, iOS 18, *)
enum CohereChunkMerge {
    private struct Token {
        let original: String
        let normalized: String
    }

    static func mergeTranscriptChunks(_ parts: [String]) -> String {
        guard let first = parts.first else { return "" }

        var accumulator = first
        for next in parts.dropFirst() {
            accumulator = self.mergeTwo(accumulator, next)
        }
        return accumulator.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func mergeTwo(_ lhs: String, _ rhs: String) -> String {
        let lhsTokens = self.tokenize(lhs)
        let rhsTokens = self.tokenize(rhs)

        if lhsTokens.isEmpty { return rhs }
        if rhsTokens.isEmpty { return lhs }

        let maxOverlap = min(lhsTokens.count, rhsTokens.count, 64)
        if let overlap = self.bestOverlap(lhsTokens: lhsTokens, rhsTokens: rhsTokens, maxOverlap: maxOverlap) {
            let remainder = rhsTokens.dropFirst(overlap).map(\.original).joined(separator: " ")
            if remainder.isEmpty {
                return lhs
            }
            return lhs + " " + remainder
        }

        return lhs + " " + rhs
    }

    private static func bestOverlap(lhsTokens: [Token], rhsTokens: [Token], maxOverlap: Int) -> Int? {
        guard maxOverlap > 0 else { return nil }

        // Prefer exact normalized overlaps first.
        for overlap in stride(from: maxOverlap, through: 2, by: -1) {
            if self.exactNormalizedMatch(lhsTokens: lhsTokens, rhsTokens: rhsTokens, overlap: overlap) {
                return overlap
            }
        }

        // Allow a single strong token overlap only for content-bearing words.
        if self.exactNormalizedMatch(lhsTokens: lhsTokens, rhsTokens: rhsTokens, overlap: 1) {
            let token = lhsTokens[lhsTokens.count - 1].normalized
            if token.count >= 5 {
                return 1
            }
        }

        // Fallback for small wording/punctuation drift across overlap.
        for overlap in stride(from: maxOverlap, through: 3, by: -1) {
            if self.fuzzyNormalizedMatch(lhsTokens: lhsTokens, rhsTokens: rhsTokens, overlap: overlap) {
                return overlap
            }
        }

        return nil
    }

    private static func exactNormalizedMatch(lhsTokens: [Token], rhsTokens: [Token], overlap: Int) -> Bool {
        let lhsSlice = lhsTokens.suffix(overlap)
        let rhsSlice = rhsTokens.prefix(overlap)
        return zip(lhsSlice, rhsSlice).allSatisfy { lhs, rhs in
            lhs.normalized.isEmpty == false &&
                rhs.normalized.isEmpty == false &&
                lhs.normalized == rhs.normalized
        }
    }

    private static func fuzzyNormalizedMatch(lhsTokens: [Token], rhsTokens: [Token], overlap: Int) -> Bool {
        let lhsSlice = Array(lhsTokens.suffix(overlap))
        let rhsSlice = Array(rhsTokens.prefix(overlap))

        var matches = 0
        var informativeMatches = 0

        for (lhs, rhs) in zip(lhsSlice, rhsSlice) {
            guard lhs.normalized.isEmpty == false, rhs.normalized.isEmpty == false else {
                continue
            }
            if lhs.normalized == rhs.normalized {
                matches += 1
                if lhs.normalized.count >= 3 {
                    informativeMatches += 1
                }
            }
        }

        let ratio = Double(matches) / Double(overlap)
        return matches >= overlap - 1 && ratio >= 0.8 && informativeMatches >= 2
    }

    private static func tokenize(_ text: String) -> [Token] {
        text
            .split(whereSeparator: \.isWhitespace)
            .map { piece in
                let original = String(piece)
                return Token(original: original, normalized: self.normalize(original))
            }
    }

    private static func normalize(_ token: String) -> String {
        let lowered = token.lowercased()
        let filtered = lowered.unicodeScalars.filter { scalar in
            CharacterSet.alphanumerics.contains(scalar) || scalar == "'" || scalar == "-"
        }
        return String(String.UnicodeScalarView(filtered))
    }
}
