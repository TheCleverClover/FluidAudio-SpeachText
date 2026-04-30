@preconcurrency import CoreML
import Foundation
import OSLog

// swiftlint:disable file_length
private let graniteAsrLogger = Logger(subsystem: "FluidAudio", category: "GraniteAsrManager")

@available(macOS 14, iOS 17, *)
public enum GraniteAsrMode: String, Sendable {
    case balanced
    case speed
    case longSpeed = "long-speed"
}

@available(macOS 14, iOS 17, *)
public struct GraniteTranscriptionResult: Sendable {
    public let text: String
    public let durationSeconds: Double
    public let elapsedSeconds: Double
    public let realTimeFactor: Double
    public let windowSeconds: Int
    public let overlapSeconds: Double
    public let chunkCount: Int
    public let overlaps: [GraniteOverlapDiagnostic]
}

@available(macOS 14, iOS 17, *)
public struct GraniteOverlapDiagnostic: Sendable {
    public let leftChunk: Int
    public let rightChunk: Int
    public let overlapStartSeconds: Double
    public let overlapEndSeconds: Double
    public let leftTokenCount: Int
    public let rightTokenCount: Int
}

@available(macOS 14, iOS 17, *)
public actor GraniteAsrManager {
    private var models: GraniteAsrModels?
    private var featureExtractor: GraniteFeatureExtractor?
    private var loadedSpeedModel: MLModel?
    private var computeUnits: MLComputeUnits = .cpuAndGPU
    private var predictionBackings: [String: GranitePredictionBacking] = [:]

    public init() {}

    public func loadModels(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws {
        let loadedModels = try await GraniteAsrModels.load(from: directory, computeUnits: computeUnits)
        let extractor = try GraniteFeatureExtractor(modelDirectory: directory, manifest: loadedModels.manifest)
        models = loadedModels
        featureExtractor = extractor
        loadedSpeedModel = loadedModels.speedModel
        self.computeUnits = computeUnits
        predictionBackings.removeAll(keepingCapacity: true)
        graniteAsrLogger.info("Granite NAR ASR models loaded")
    }

    public func transcribe(
        audioSamples: [Float],
        mode: GraniteAsrMode = .balanced
    ) async throws -> String {
        let result = try await transcribeDetailed(audioSamples: audioSamples, mode: mode)
        return result.text
    }

    public func transcribeDetailed(
        audioSamples: [Float],
        mode: GraniteAsrMode = .balanced
    ) async throws -> GraniteTranscriptionResult {
        guard let models else {
            throw GraniteAsrError.invalidOutput("Granite models are not loaded")
        }
        guard let featureExtractor else {
            throw GraniteAsrError.invalidOutput("Granite feature extractor is not loaded")
        }
        guard !audioSamples.isEmpty else {
            throw GraniteAsrError.invalidAudio("Audio is empty")
        }

        let durationSeconds = Double(audioSamples.count) / Double(models.manifest.sampleRate)
        let (windowSeconds, overlapSeconds) = settings(
            for: mode,
            durationSeconds: durationSeconds,
            manifest: models.manifest
        )
        let selected = try await selectedModel(windowSeconds: windowSeconds, models: models)
        let chunks = try makeChunks(
            totalSamples: audioSamples.count,
            sampleRate: models.manifest.sampleRate,
            windowSeconds: selected.meta.seconds,
            overlapSeconds: overlapSeconds,
            minTailSeconds: 1.0
        )

        graniteAsrLogger.info("Granite transcribe start [duration=\(durationSeconds, privacy: .public)]")
        graniteAsrLogger.info(
            "Granite chunking [window=\(selected.meta.seconds), overlap=\(overlapSeconds), chunks=\(chunks.count)]"
        )

        let startedAt = CFAbsoluteTimeGetCurrent()
        let decodedChunks = try decodeChunks(
            audioSamples: audioSamples,
            chunks: chunks,
            selected: selected,
            models: models,
            featureExtractor: featureExtractor
        )
        let mergedSpans = alignmentMergedSpans(chunks: chunks, decoded: decodedChunks)
        let mergedText = models.tokenizer.decode(mergedSpans.map(\.decodeID))
        let elapsedSeconds = CFAbsoluteTimeGetCurrent() - startedAt
        let overlaps = overlapDiagnostics(chunks: chunks, decoded: decodedChunks)

        return GraniteTranscriptionResult(
            text: mergedText,
            durationSeconds: durationSeconds,
            elapsedSeconds: elapsedSeconds,
            realTimeFactor: elapsedSeconds / max(durationSeconds, 1e-6),
            windowSeconds: selected.meta.seconds,
            overlapSeconds: overlapSeconds,
            chunkCount: chunks.count,
            overlaps: overlaps
        )
    }

    private func decodeChunks(
        audioSamples: [Float],
        chunks: [GraniteChunkWindow],
        selected: (model: MLModel, meta: GraniteWindowMeta),
        models: GraniteAsrModels,
        featureExtractor: GraniteFeatureExtractor
    ) throws -> [[GraniteTokenSpan]] {
        var decodedChunks: [[GraniteTokenSpan]] = []
        decodedChunks.reserveCapacity(chunks.count)

        for chunk in chunks {
            let window = try featureExtractor.makeWindow(
                audio: audioSamples[chunk.startSample ..< chunk.endSample],
                windowFrames: selected.meta.frames
            )
            let outputs = try runGreedyModel(model: selected.model, meta: selected.meta, window: window)
            let spans = decodeTokenSpans(
                tokenArray: outputs.tokenArray,
                bpeLength: outputs.bpeLength,
                chunk: chunk,
                manifest: models.manifest
            )
            decodedChunks.append(spans)
        }

        return decodedChunks
    }

    private func selectedModel(
        windowSeconds: Int,
        models: GraniteAsrModels
    ) async throws -> (model: MLModel, meta: GraniteWindowMeta) {
        let key = "\(windowSeconds)s"
        guard let meta = models.manifest.windows[key] else {
            throw GraniteAsrError.modelNotFound("window \(key)")
        }

        if windowSeconds == models.manifest.defaultWindowSeconds {
            return (models.balancedModel, meta)
        }
        if windowSeconds == models.manifest.speedWindowSeconds, let speedModel = loadedSpeedModel ?? models.speedModel {
            return (speedModel, meta)
        }
        if windowSeconds == models.manifest.speedWindowSeconds {
            let speedModel = try await GraniteAsrModels.loadWindowModel(
                meta,
                from: models.modelDirectory,
                computeUnits: computeUnits
            )
            loadedSpeedModel = speedModel
            return (speedModel, meta)
        }
        if let fallbackMeta = models.manifest.windows["\(models.manifest.defaultWindowSeconds)s"] {
            return (models.balancedModel, fallbackMeta)
        }
        throw GraniteAsrError.modelNotFound("window \(key)")
    }

    private func runGreedyModel(
        model: MLModel,
        meta: GraniteWindowMeta,
        window: GraniteFeatureWindow
    ) throws -> (tokenArray: MLMultiArray, bpeLength: Int) {
        let inputName = meta.inputs.first ?? "input_features"
        let maskName = meta.inputs.dropFirst().first ?? "attention_mask"
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: window.inputFeatures),
            maskName: MLFeatureValue(multiArray: window.attentionMask)
        ])
        let backing = try predictionBacking(for: meta)

        let prediction = try model.prediction(from: provider, options: backing.options)
        guard let tokenArray = prediction.featureValue(for: backing.tokenOutputName)?.multiArrayValue else {
            throw GraniteAsrError.invalidOutput("Missing \(backing.tokenOutputName)")
        }
        guard let lengthArray = prediction.featureValue(for: backing.lengthOutputName)?.multiArrayValue else {
            throw GraniteAsrError.invalidOutput("Missing \(backing.lengthOutputName)")
        }

        let lengthPtr = lengthArray.dataPointer.bindMemory(to: Int32.self, capacity: lengthArray.count)
        let bpeLength = min(Int(lengthPtr[0]), meta.bpeFrames)
        return (tokenArray, max(0, bpeLength))
    }

    private func predictionBacking(for meta: GraniteWindowMeta) throws -> GranitePredictionBacking {
        if let backing = predictionBackings[meta.package] {
            return backing
        }

        let tokenOutput = meta.outputs.first ?? "bpe_token_ids"
        let lengthOutput = meta.outputs.dropFirst().first ?? "bpe_length"
        let tokenIDs = try MLMultiArray(
            shape: [1, NSNumber(value: meta.bpeFrames)],
            dataType: .int32
        )
        let length = try MLMultiArray(shape: [1], dataType: .int32)
        let options = MLPredictionOptions()
        options.outputBackings = [
            tokenOutput: tokenIDs,
            lengthOutput: length
        ]

        let backing = GranitePredictionBacking(
            tokenOutputName: tokenOutput,
            lengthOutputName: lengthOutput,
            options: options
        )
        predictionBackings[meta.package] = backing
        return backing
    }

    private func decodeTokenSpans(
        tokenArray: MLMultiArray,
        bpeLength: Int,
        chunk: GraniteChunkWindow,
        manifest: GraniteAsrManifest
    ) -> [GraniteTokenSpan] {
        guard bpeLength > 0 else { return [] }

        let limit = min(bpeLength, tokenArray.count)
        let tokenIDs = tokenArray.dataPointer.bindMemory(to: Int32.self, capacity: tokenArray.count)
        var spans: [GraniteTokenSpan] = []
        var runStart = 0

        for index in 1 ... limit {
            if index < limit, tokenIDs[index] == tokenIDs[runStart] {
                continue
            }

            let tokenID = Int(tokenIDs[runStart])
            if tokenID != 0 {
                let decodeID = tokenID - 1
                let frameRate = Double(manifest.featureFramesPerSecond)
                let startSeconds = chunk.startSeconds + Double(runStart * manifest.bpePoolingWindow) / frameRate
                let endSeconds = chunk.startSeconds + Double(index * manifest.bpePoolingWindow) / frameRate
                spans.append(
                    GraniteTokenSpan(
                        tokenID: tokenID,
                        decodeID: decodeID,
                        startSeconds: startSeconds,
                        endSeconds: endSeconds,
                        chunkIndex: chunk.index
                    )
                )
            }
            runStart = index
        }

        return spans
    }
}

@available(macOS 14, iOS 17, *)
private struct GraniteChunkWindow: Sendable {
    let index: Int
    let startSample: Int
    let endSample: Int
    let windowSamples: Int
    let sampleRate: Int

    var startSeconds: Double {
        Double(startSample) / Double(sampleRate)
    }

    var endSeconds: Double {
        Double(endSample) / Double(sampleRate)
    }

    var durationSeconds: Double {
        Double(endSample - startSample) / Double(sampleRate)
    }
}

@available(macOS 14, iOS 17, *)
private struct GranitePredictionBacking {
    let tokenOutputName: String
    let lengthOutputName: String
    let options: MLPredictionOptions
}

@available(macOS 14, iOS 17, *)
private struct GraniteTokenSpan: Sendable {
    let tokenID: Int
    let decodeID: Int
    let startSeconds: Double
    let endSeconds: Double
    let chunkIndex: Int

    var centerSeconds: Double {
        0.5 * (startSeconds + endSeconds)
    }
}

@available(macOS 14, iOS 17, *)
private func settings(
    for mode: GraniteAsrMode,
    durationSeconds: Double,
    manifest: GraniteAsrManifest
) -> (Int, Double) {
    switch mode {
    case .balanced:
        return (manifest.defaultWindowSeconds, manifest.defaultOverlapSeconds)
    case .speed:
        if durationSeconds <= 300.0 {
            return (manifest.defaultWindowSeconds, manifest.defaultOverlapSeconds)
        }
        return (manifest.speedWindowSeconds, manifest.speedOverlapSeconds)
    case .longSpeed:
        return (manifest.speedWindowSeconds, manifest.speedOverlapSeconds)
    }
}

@available(macOS 14, iOS 17, *)
private func makeChunks(
    totalSamples: Int,
    sampleRate: Int,
    windowSeconds: Int,
    overlapSeconds: Double,
    minTailSeconds: Double
) throws -> [GraniteChunkWindow] {
    let windowSamples = windowSeconds * sampleRate
    let overlapSamples = Int((overlapSeconds * Double(sampleRate)).rounded())
    let minTailSamples = Int((minTailSeconds * Double(sampleRate)).rounded())
    let strideSamples = windowSamples - overlapSamples

    guard totalSamples > 0 else {
        throw GraniteAsrError.invalidAudio("Audio is empty")
    }
    guard overlapSamples >= 0, strideSamples > 0 else {
        throw GraniteAsrError.invalidAudio("Overlap must be smaller than window")
    }

    var chunks: [GraniteChunkWindow] = []
    var start = 0
    while start < totalSamples {
        let end = min(start + windowSamples, totalSamples)
        chunks.append(
            GraniteChunkWindow(
                index: chunks.count,
                startSample: start,
                endSample: end,
                windowSamples: windowSamples,
                sampleRate: sampleRate
            )
        )
        if end == totalSamples || totalSamples - end <= minTailSamples {
            break
        }
        start += strideSamples
    }
    return chunks
}

@available(macOS 14, iOS 17, *)
private func alignmentMergedSpans(
    chunks: [GraniteChunkWindow],
    decoded: [[GraniteTokenSpan]]
) -> [GraniteTokenSpan] {
    guard chunks.count == decoded.count, !chunks.isEmpty else { return [] }

    var seams: [(start: Double, end: Double)] = []
    var seamSpans: [[GraniteTokenSpan]] = []
    for index in 0 ..< (chunks.count - 1) {
        let leftChunk = chunks[index]
        let rightChunk = chunks[index + 1]
        let overlapStart = max(leftChunk.startSeconds, rightChunk.startSeconds)
        let overlapEnd = min(leftChunk.endSeconds, rightChunk.endSeconds)
        guard overlapEnd > overlapStart else { continue }

        let left = decoded[index].filter { overlapStart <= $0.centerSeconds && $0.centerSeconds < overlapEnd }
        let right = decoded[index + 1].filter { overlapStart <= $0.centerSeconds && $0.centerSeconds < overlapEnd }
        seams.append((overlapStart, overlapEnd))
        seamSpans.append(fuseOverlap(left: left, right: right, leftChunk: leftChunk, rightChunk: rightChunk))
    }

    var merged: [GraniteTokenSpan] = []
    for index in chunks.indices {
        let coreStart = index == 0 ? -Double.infinity : seams[index - 1].end
        let coreEnd = index == chunks.count - 1 ? Double.infinity : seams[index].start
        merged.append(
            contentsOf: decoded[index].filter { coreStart <= $0.centerSeconds && $0.centerSeconds < coreEnd }
        )
        if index < seamSpans.count {
            merged.append(contentsOf: seamSpans[index])
        }
    }

    return merged.sorted { left, right in
        if left.centerSeconds == right.centerSeconds {
            return left.chunkIndex < right.chunkIndex
        }
        return left.centerSeconds < right.centerSeconds
    }
}

@available(macOS 14, iOS 17, *)
private func fuseOverlap(
    left: [GraniteTokenSpan],
    right: [GraniteTokenSpan],
    leftChunk: GraniteChunkWindow,
    rightChunk: GraniteChunkWindow
) -> [GraniteTokenSpan] {
    if left.isEmpty { return right }
    if right.isEmpty { return left }

    let anchors = lcsAnchors(left.map(\.decodeID), right.map(\.decodeID))
    var fused: [GraniteTokenSpan] = []
    var leftCursor = 0
    var rightCursor = 0

    func appendGap(leftEnd: Int, rightEnd: Int) {
        let leftPart = Array(left[leftCursor ..< leftEnd])
        let rightPart = Array(right[rightCursor ..< rightEnd])
        if leftPart.isEmpty, !rightPart.isEmpty {
            if averageQuality(rightPart, chunk: rightChunk) >= 0.20 {
                fused.append(contentsOf: rightPart)
            }
        } else if rightPart.isEmpty, !leftPart.isEmpty {
            if averageQuality(leftPart, chunk: leftChunk) >= 0.20 {
                fused.append(contentsOf: leftPart)
            }
        } else if !leftPart.isEmpty || !rightPart.isEmpty {
            let leftScore = averageQuality(leftPart, chunk: leftChunk)
            let rightScore = averageQuality(rightPart, chunk: rightChunk)
            fused.append(contentsOf: rightScore > leftScore ? rightPart : leftPart)
        }
    }

    for anchor in anchors {
        appendGap(leftEnd: anchor.left, rightEnd: anchor.right)
        let leftSpan = left[anchor.left]
        let rightSpan = right[anchor.right]
        let selectedSpan = edgeQuality(rightSpan, chunk: rightChunk) > edgeQuality(leftSpan, chunk: leftChunk)
            ? rightSpan
            : leftSpan
        fused.append(selectedSpan)
        leftCursor = anchor.left + 1
        rightCursor = anchor.right + 1
    }

    appendGap(leftEnd: left.count, rightEnd: right.count)
    return fused.sorted { lhs, rhs in
        if lhs.centerSeconds == rhs.centerSeconds {
            return lhs.chunkIndex < rhs.chunkIndex
        }
        return lhs.centerSeconds < rhs.centerSeconds
    }
}

private func lcsAnchors(_ left: [Int], _ right: [Int]) -> [(left: Int, right: Int)] {
    guard !left.isEmpty, !right.isEmpty else { return [] }

    let columnCount = right.count + 1
    var lengths = [Int](repeating: 0, count: (left.count + 1) * columnCount)
    for leftIndex in stride(from: left.count - 1, through: 0, by: -1) {
        for rightIndex in stride(from: right.count - 1, through: 0, by: -1) {
            if left[leftIndex] == right[rightIndex] {
                lengths[leftIndex * columnCount + rightIndex] =
                    lengths[(leftIndex + 1) * columnCount + (rightIndex + 1)] + 1
            } else {
                lengths[leftIndex * columnCount + rightIndex] = max(
                    lengths[(leftIndex + 1) * columnCount + rightIndex],
                    lengths[leftIndex * columnCount + (rightIndex + 1)]
                )
            }
        }
    }

    var anchors: [(left: Int, right: Int)] = []
    var leftIndex = 0
    var rightIndex = 0
    while leftIndex < left.count, rightIndex < right.count {
        if left[leftIndex] == right[rightIndex] {
            anchors.append((leftIndex, rightIndex))
            leftIndex += 1
            rightIndex += 1
        } else if lengths[(leftIndex + 1) * columnCount + rightIndex] >=
            lengths[leftIndex * columnCount + (rightIndex + 1)] {
            leftIndex += 1
        } else {
            rightIndex += 1
        }
    }
    return anchors
}

@available(macOS 14, iOS 17, *)
private func edgeQuality(_ span: GraniteTokenSpan, chunk: GraniteChunkWindow) -> Double {
    let edge = min(span.centerSeconds - chunk.startSeconds, chunk.endSeconds - span.centerSeconds)
    return max(0.0, edge) / max(chunk.durationSeconds * 0.5, 1e-6)
}

@available(macOS 14, iOS 17, *)
private func averageQuality(_ spans: [GraniteTokenSpan], chunk: GraniteChunkWindow) -> Double {
    guard !spans.isEmpty else { return 0.0 }
    return spans.reduce(0.0) { $0 + edgeQuality($1, chunk: chunk) } / Double(spans.count)
}

@available(macOS 14, iOS 17, *)
private func overlapDiagnostics(
    chunks: [GraniteChunkWindow],
    decoded: [[GraniteTokenSpan]]
) -> [GraniteOverlapDiagnostic] {
    guard chunks.count == decoded.count, chunks.count > 1 else { return [] }

    var diagnostics: [GraniteOverlapDiagnostic] = []
    for index in 0 ..< (chunks.count - 1) {
        let leftChunk = chunks[index]
        let rightChunk = chunks[index + 1]
        let overlapStart = max(leftChunk.startSeconds, rightChunk.startSeconds)
        let overlapEnd = min(leftChunk.endSeconds, rightChunk.endSeconds)
        let leftCount = decoded[index].filter {
            overlapStart <= $0.centerSeconds && $0.centerSeconds < overlapEnd
        }.count
        let rightCount = decoded[index + 1].filter {
            overlapStart <= $0.centerSeconds && $0.centerSeconds < overlapEnd
        }.count
        diagnostics.append(
            GraniteOverlapDiagnostic(
                leftChunk: leftChunk.index,
                rightChunk: rightChunk.index,
                overlapStartSeconds: overlapStart,
                overlapEndSeconds: overlapEnd,
                leftTokenCount: leftCount,
                rightTokenCount: rightCount
            )
        )
    }
    return diagnostics
}
