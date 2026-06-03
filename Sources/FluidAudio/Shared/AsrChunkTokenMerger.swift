import Foundation

/// Model-agnostic overlap merge for token windows produced by chunked ASR.
///
/// This is the merge strategy that has shipped in `ChunkProcessor` (Parakeet TDT) since the
/// long-audio chunking work: align tokens in the overlap region (contiguous run first, then
/// LCS), splice the windows, and fall back to a timestamp midpoint cut when alignment fails.
///
/// It only needs a token id + an encoder-frame timestamp per token, so it is reused verbatim by
/// both the Parakeet TDT path and the Nemotron RNNT offline path. Confidence/duration are carried
/// through untouched.
enum AsrChunkTokenMerger {
    /// (token id, encoder-frame index, confidence, duration). `duration` is decoder-specific
    /// (TDT only) and is preserved but ignored by the merge.
    typealias TokenWindow = (token: Int, timestamp: Int, confidence: Float, duration: Int)

    private struct IndexedToken {
        let index: Int
        let token: TokenWindow
        let start: Double
        let end: Double
    }

    /// Merge a sequence of per-window token lists (in chunk order) into one stream.
    static func merge(
        _ chunks: [[TokenWindow]],
        sampleRate: Int,
        overlapSeconds: Double
    ) -> [TokenWindow] {
        guard var merged = chunks.first else { return [] }
        for chunk in chunks.dropFirst() {
            merged = mergePair(merged, chunk, sampleRate: sampleRate, overlapSeconds: overlapSeconds)
        }
        return merged
    }

    /// Merge two adjacent windows (`left` precedes `right`).
    static func mergePair(
        _ left: [TokenWindow],
        _ right: [TokenWindow],
        sampleRate: Int,
        overlapSeconds: Double
    ) -> [TokenWindow] {
        if left.isEmpty { return right }
        if right.isEmpty { return left }

        let frameDuration = Double(ASRConstants.samplesPerEncoderFrame) / Double(sampleRate)
        let halfOverlapWindow = overlapSeconds / 2

        func startTime(of token: TokenWindow) -> Double {
            Double(token.timestamp) * frameDuration
        }

        func endTime(of token: TokenWindow) -> Double {
            startTime(of: token) + frameDuration
        }

        let leftEndTime = endTime(of: left.last!)
        let rightStartTime = startTime(of: right.first!)

        if leftEndTime <= rightStartTime {
            return left + right
        }

        let overlapLeft: [IndexedToken] = left.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            let end = start + frameDuration
            guard end > rightStartTime - overlapSeconds else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: end)
        }

        let overlapRight: [IndexedToken] = right.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            guard start < leftEndTime + overlapSeconds else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: start + frameDuration)
        }

        guard overlapLeft.count >= 2 && overlapRight.count >= 2 else {
            return mergeByMidpoint(
                left: left, right: right, leftEndTime: leftEndTime, rightStartTime: rightStartTime,
                frameDuration: frameDuration)
        }

        let minimumPairs = max(overlapLeft.count / 2, 1)

        let contiguousPairs = findBestContiguousPairs(
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            tolerance: halfOverlapWindow
        )

        if contiguousPairs.count >= minimumPairs {
            return mergeUsingMatches(
                matches: contiguousPairs,
                overlapLeft: overlapLeft,
                overlapRight: overlapRight,
                left: left,
                right: right
            )
        }

        let lcsPairs = findLongestCommonSubsequencePairs(
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            tolerance: halfOverlapWindow
        )

        guard !lcsPairs.isEmpty else {
            return mergeByMidpoint(
                left: left, right: right, leftEndTime: leftEndTime, rightStartTime: rightStartTime,
                frameDuration: frameDuration)
        }

        return mergeUsingMatches(
            matches: lcsPairs,
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            left: left,
            right: right
        )
    }

    private static func findBestContiguousPairs(
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        tolerance: Double
    ) -> [(Int, Int)] {
        var best: [(Int, Int)] = []

        for i in 0..<overlapLeft.count {
            for j in 0..<overlapRight.count {
                let leftToken = overlapLeft[i]
                let rightToken = overlapRight[j]

                if tokensMatch(leftToken, rightToken, tolerance: tolerance) {
                    var current: [(Int, Int)] = []
                    var k = i
                    var l = j

                    while k < overlapLeft.count && l < overlapRight.count {
                        let nextLeft = overlapLeft[k]
                        let nextRight = overlapRight[l]

                        if tokensMatch(nextLeft, nextRight, tolerance: tolerance) {
                            current.append((k, l))
                            k += 1
                            l += 1
                        } else {
                            break
                        }
                    }

                    if current.count > best.count {
                        best = current
                    }
                }
            }
        }

        return best
    }

    private static func findLongestCommonSubsequencePairs(
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        tolerance: Double
    ) -> [(Int, Int)] {
        let leftCount = overlapLeft.count
        let rightCount = overlapRight.count

        var dp = Array(repeating: Array(repeating: 0, count: rightCount + 1), count: leftCount + 1)

        for i in 1...leftCount {
            for j in 1...rightCount {
                if tokensMatch(overlapLeft[i - 1], overlapRight[j - 1], tolerance: tolerance) {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        var pairs: [(Int, Int)] = []
        var i = leftCount
        var j = rightCount

        while i > 0 && j > 0 {
            if tokensMatch(overlapLeft[i - 1], overlapRight[j - 1], tolerance: tolerance) {
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }

        return pairs.reversed()
    }

    private static func tokensMatch(_ left: IndexedToken, _ right: IndexedToken, tolerance: Double) -> Bool {
        guard left.token.token == right.token.token else { return false }
        let timeDifference = abs(left.start - right.start)
        return timeDifference < tolerance
    }

    private static func mergeUsingMatches(
        matches: [(Int, Int)],
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        left: [TokenWindow],
        right: [TokenWindow]
    ) -> [TokenWindow] {
        let leftIndices = matches.map { overlapLeft[$0.0].index }
        let rightIndices = matches.map { overlapRight[$0.1].index }

        var result: [TokenWindow] = []

        if let firstLeft = leftIndices.first, firstLeft > 0 {
            result.append(contentsOf: left[..<firstLeft])
        }

        for idx in 0..<matches.count {
            let leftIndex = leftIndices[idx]
            let rightIndex = rightIndices[idx]

            result.append(left[leftIndex])

            guard idx < matches.count - 1 else { continue }

            let nextLeftIndex = leftIndices[idx + 1]
            let nextRightIndex = rightIndices[idx + 1]

            let gapLeft = nextLeftIndex > leftIndex + 1 ? Array(left[(leftIndex + 1)..<nextLeftIndex]) : []
            let gapRight = nextRightIndex > rightIndex + 1 ? Array(right[(rightIndex + 1)..<nextRightIndex]) : []

            if gapRight.count > gapLeft.count {
                result.append(contentsOf: gapRight)
            } else {
                result.append(contentsOf: gapLeft)
            }
        }

        if let lastRight = rightIndices.last, lastRight + 1 < right.count {
            result.append(contentsOf: right[(lastRight + 1)...])
        }

        return result
    }

    private static func mergeByMidpoint(
        left: [TokenWindow],
        right: [TokenWindow],
        leftEndTime: Double,
        rightStartTime: Double,
        frameDuration: Double
    ) -> [TokenWindow] {
        let cutoff = (leftEndTime + rightStartTime) / 2
        let trimmedLeft = left.filter { Double($0.timestamp) * frameDuration <= cutoff }
        let trimmedRight = right.filter { Double($0.timestamp) * frameDuration >= cutoff }
        return trimmedLeft + trimmedRight
    }
}
