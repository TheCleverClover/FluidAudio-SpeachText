import Foundation

@available(macOS 15, iOS 18, *)
enum CohereAudioChunkPlanner {
    static let defaultBoundarySearchSeconds = 5.0
    static let minEnergyWindowSamples = 1_600

    struct Chunk: Equatable {
        let startSample: Int
        let endSample: Int

        var range: Range<Int> { self.startSample..<self.endSample }
        var sampleCount: Int { self.endSample - self.startSample }
    }

    static func makeChunks(
        audioSamples: [Float],
        sampleRate: Int,
        maxAudioSamples: Int,
        boundarySearchSeconds: Double = defaultBoundarySearchSeconds,
        minEnergyWindowSamples: Int = minEnergyWindowSamples
    ) -> [Chunk] {
        guard audioSamples.isEmpty == false, sampleRate > 0, maxAudioSamples > 0 else {
            return []
        }

        let totalSamples = audioSamples.count
        // Mirror the HF processor's long-audio strategy: keep a short fast path,
        // then search for a quiet split point near the end of each model-sized window.
        let boundaryContextSize = max(
            1,
            min(
                Int(round(boundarySearchSeconds * Double(sampleRate))),
                maxAudioSamples - 1
            )
        )
        let fastPathThreshold = max(1, maxAudioSamples - boundaryContextSize)

        if totalSamples <= fastPathThreshold {
            return [Chunk(startSample: 0, endSample: totalSamples)]
        }

        var chunks: [Chunk] = []
        chunks.reserveCapacity(max(1, Int(ceil(Double(totalSamples) / Double(fastPathThreshold)))))

        var currentStart = 0
        while currentStart < totalSamples {
            if currentStart + maxAudioSamples >= totalSamples {
                chunks.append(Chunk(startSample: currentStart, endSample: totalSamples))
                break
            }

            let searchStart = max(currentStart, currentStart + maxAudioSamples - boundaryContextSize)
            let searchEnd = min(currentStart + maxAudioSamples, totalSamples)
            let splitPoint: Int
            if searchEnd <= searchStart {
                splitPoint = currentStart + maxAudioSamples
            } else {
                splitPoint = self.findQuietestSplitPoint(
                    audioSamples,
                    startSample: searchStart,
                    endSample: searchEnd,
                    minEnergyWindowSamples: minEnergyWindowSamples
                )
            }

            let boundedSplit = max(currentStart + 1, min(splitPoint, totalSamples))
            chunks.append(Chunk(startSample: currentStart, endSample: boundedSplit))
            currentStart = boundedSplit
        }

        return chunks
    }

    static func joinChunkTexts(_ parts: [String]) -> String {
        var pieces = parts
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { $0.isEmpty == false }

        guard var accumulator = pieces.first else { return "" }
        pieces.removeFirst()

        for next in pieces {
            if self.shouldInsertSpace(between: accumulator, and: next) {
                accumulator += " "
            }
            accumulator += next
        }

        return accumulator.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func findQuietestSplitPoint(
        _ audioSamples: [Float],
        startSample: Int,
        endSample: Int,
        minEnergyWindowSamples: Int = minEnergyWindowSamples
    ) -> Int {
        let boundedStart = max(0, min(startSample, audioSamples.count))
        let boundedEnd = max(boundedStart, min(endSample, audioSamples.count))
        let sampleCount = boundedEnd - boundedStart

        if sampleCount <= minEnergyWindowSamples {
            return (boundedStart + boundedEnd) / 2
        }

        var quietestIndex = boundedStart
        var minimumEnergy = Float.greatestFiniteMagnitude
        let upperBound = sampleCount - minEnergyWindowSamples
        var offset = 0

        while offset <= upperBound {
            let windowStart = boundedStart + offset
            let windowEnd = windowStart + minEnergyWindowSamples
            var energyAccumulator: Float = 0
            for sample in audioSamples[windowStart..<windowEnd] {
                energyAccumulator += sample * sample
            }
            let rms = sqrt(energyAccumulator / Float(minEnergyWindowSamples))
            if rms < minimumEnergy {
                minimumEnergy = rms
                quietestIndex = windowStart
            }
            offset += minEnergyWindowSamples
        }

        return quietestIndex
    }

    private static func shouldInsertSpace(between lhs: String, and rhs: String) -> Bool {
        guard
            let lastScalar = lhs.unicodeScalars.last,
            let firstScalar = rhs.unicodeScalars.first
        else {
            return false
        }

        if CharacterSet.whitespacesAndNewlines.contains(lastScalar) {
            return false
        }

        if self.isCJK(lastScalar) || self.isCJK(firstScalar) {
            return false
        }

        if CharacterSet.punctuationCharacters.contains(firstScalar) {
            return false
        }

        return true
    }

    private static func isCJK(_ scalar: UnicodeScalar) -> Bool {
        switch scalar.value {
        case 0x3040...0x30FF, 0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF:
            return true
        default:
            return false
        }
    }
}
