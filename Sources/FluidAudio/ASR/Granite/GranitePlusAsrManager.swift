@preconcurrency import CoreML
import Foundation
import OSLog

private let granitePlusAsrLogger = Logger(subsystem: "FluidAudio", category: "GranitePlusAsrManager")

@available(macOS 15, iOS 18, *)
public struct GranitePlusTranscriptionResult: Sendable {
    public let text: String
    public let rawText: String
    public let durationSeconds: Double
    public let elapsedSeconds: Double
    public let realTimeFactor: Double
    public let windowSeconds: Int
    public let overlapSeconds: Double
    public let chunkCount: Int
    public let generatedTokens: Int
    public let stoppedOnEOS: Bool
    public let task: GranitePlusTask
}

@available(macOS 15, iOS 18, *)
private struct GranitePlusPreparedPrompt {
    let embeddings: MLMultiArray
    let tokenCount: Int
}

@available(macOS 15, iOS 18, *)
private struct GranitePlusChunkResult {
    let text: String
    let rawText: String
    let generatedTokens: Int
    let stoppedOnEOS: Bool
}

@available(macOS 15, iOS 18, *)
public actor GranitePlusAsrManager {
    private static let defaultOverlapSeconds = 5.0
    private static let minTailSeconds = 1.0

    private var models: GranitePlusAsrModels?
    private var featureExtractor: GraniteFeatureExtractor?
    private var positionCache: [Int: MLMultiArray] = [:]
    private var causalMaskCache: [String: MLMultiArray] = [:]

    public init() {}

    public func loadModels(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws {
        let loadedModels = try await GranitePlusAsrModels.load(from: directory, computeUnits: computeUnits)
        let extractor = try GraniteFeatureExtractor(
            modelDirectory: loadedModels.modelDirectory,
            manifest: loadedModels.featureManifest()
        )
        models = loadedModels
        featureExtractor = extractor
        positionCache.removeAll(keepingCapacity: true)
        causalMaskCache.removeAll(keepingCapacity: true)
        granitePlusAsrLogger.info("Granite Plus ASR models loaded")
    }

    public func transcribe(
        audioSamples: [Float],
        task: GranitePlusTask = .asr,
        maxNewTokens: Int = 256
    ) async throws -> String {
        let result = try await self.transcribeDetailed(
            audioSamples: audioSamples,
            task: task,
            maxNewTokens: maxNewTokens
        )
        return result.text
    }

    public func transcribeDetailed(
        audioSamples: [Float],
        task: GranitePlusTask = .asr,
        maxNewTokens: Int = 256
    ) async throws -> GranitePlusTranscriptionResult {
        guard let models else {
            throw GraniteAsrError.invalidOutput("Granite Plus models are not loaded")
        }
        guard let featureExtractor else {
            throw GraniteAsrError.invalidOutput("Granite Plus feature extractor is not loaded")
        }
        guard !audioSamples.isEmpty else {
            throw GraniteAsrError.invalidAudio("Audio is empty")
        }

        let startedAt = CFAbsoluteTimeGetCurrent()
        let manifest = models.manifest
        let windowKey = "\(manifest.defaultWindowSeconds)s"
        guard let windowMeta = manifest.windows[windowKey] else {
            throw GraniteAsrError.modelNotFound("Granite Plus window \(windowKey)")
        }
        let chunks = try makePlusChunks(
            totalSamples: audioSamples.count,
            sampleRate: manifest.sampleRate,
            windowSeconds: windowMeta.seconds,
            overlapSeconds: Self.defaultOverlapSeconds,
            minTailSeconds: Self.minTailSeconds
        )

        var chunkResults: [GranitePlusChunkResult] = []
        chunkResults.reserveCapacity(chunks.count)

        for chunk in chunks {
            let result = try await self.transcribeChunk(
                audioSamples: audioSamples[chunk.startSample..<chunk.endSample],
                task: task,
                maxNewTokens: maxNewTokens,
                models: models,
                featureExtractor: featureExtractor,
                windowMeta: windowMeta
            )
            chunkResults.append(result)
        }

        let rawText: String
        let text: String
        if task == .timestamp {
            let timestampChunks = zip(chunks, chunkResults).map { chunk, result in
                GranitePlusTimestampParser.parse(result.rawText, chunkOffsetSeconds: chunk.startSeconds)
            }
            let mergedTokens = GranitePlusTimestampParser.merge(timestampChunks)
            rawText =
                mergedTokens
                .map { "\($0.text) [T:\(Int(($0.endSeconds * 100).rounded()))]" }
                .joined(separator: " ")
            text = mergedTokens.map(\.text).joined(separator: " ")
        } else {
            text = GranitePlusChunkMerge.mergeTranscriptChunks(chunkResults.map(\.text))
            rawText = GranitePlusChunkMerge.mergeTranscriptChunks(chunkResults.map(\.rawText))
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - startedAt
        let durationSeconds = Double(audioSamples.count) / Double(manifest.sampleRate)
        return GranitePlusTranscriptionResult(
            text: text,
            rawText: rawText,
            durationSeconds: durationSeconds,
            elapsedSeconds: elapsed,
            realTimeFactor: elapsed / max(durationSeconds, 1e-6),
            windowSeconds: windowMeta.seconds,
            overlapSeconds: chunks.count > 1 ? Self.defaultOverlapSeconds : 0,
            chunkCount: chunks.count,
            generatedTokens: chunkResults.reduce(0) { $0 + $1.generatedTokens },
            stoppedOnEOS: chunkResults.allSatisfy(\.stoppedOnEOS),
            task: task
        )
    }

    public func transcribe(
        audioFileAt url: URL,
        task: GranitePlusTask = .asr,
        maxNewTokens: Int = 256
    ) async throws -> String {
        let result = try await self.transcribeDetailed(
            audioFileAt: url,
            task: task,
            maxNewTokens: maxNewTokens
        )
        return result.text
    }

    public func transcribeDetailed(
        audioFileAt url: URL,
        task: GranitePlusTask = .asr,
        maxNewTokens: Int = 256
    ) async throws -> GranitePlusTranscriptionResult {
        guard let models else {
            throw GraniteAsrError.invalidOutput("Granite Plus models are not loaded")
        }
        let audioConverter = AudioConverter(sampleRate: Double(models.manifest.sampleRate))
        let audioSamples = try audioConverter.resampleAudioFile(url)
        return try await self.transcribeDetailed(
            audioSamples: audioSamples,
            task: task,
            maxNewTokens: maxNewTokens
        )
    }

    private func transcribeChunk<C>(
        audioSamples: C,
        task: GranitePlusTask,
        maxNewTokens: Int,
        models: GranitePlusAsrModels,
        featureExtractor: GraniteFeatureExtractor,
        windowMeta: GranitePlusAudioWindowMeta
    ) async throws -> GranitePlusChunkResult where C: RandomAccessCollection, C.Element == Float {
        let preparedPrompt = try preparePrompt(
            audioSamples: audioSamples,
            task: task,
            models: models,
            featureExtractor: featureExtractor,
            windowMeta: windowMeta
        )
        let state = models.languageModel.makeState()
        var logits = try await runLanguageModel(
            model: models.languageModel,
            state: state,
            embeddings: preparedPrompt.embeddings,
            startPosition: 0
        )
        let decoded = try await generate(
            logits: &logits,
            state: state,
            startPosition: preparedPrompt.tokenCount,
            maxNewTokens: maxNewTokens,
            models: models
        )

        let rawText = models.tokenizer.decode(decoded.tokenIDs, lowercased: false)
        let text =
            task == .timestamp
            ? GranitePlusTimestampParser.plainText(from: rawText)
            : rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        return GranitePlusChunkResult(
            text: text,
            rawText: rawText,
            generatedTokens: decoded.tokenIDs.count,
            stoppedOnEOS: decoded.stoppedOnEOS
        )
    }

    private func preparePrompt(
        audioSamples: some RandomAccessCollection<Float>,
        task: GranitePlusTask,
        models: GranitePlusAsrModels,
        featureExtractor: GraniteFeatureExtractor,
        windowMeta: GranitePlusAudioWindowMeta
    ) throws -> GranitePlusPreparedPrompt {
        let window = try featureExtractor.makeWindow(audio: audioSamples, windowFrames: windowMeta.frames)
        let audioEmbeds = try runAudioModel(models: models, meta: windowMeta, features: window.inputFeatures)
        let audioTokenCount = validAudioTokenCount(validEncoderFrames: window.validEncoderFrames, meta: windowMeta)
        let tokenIDs = try promptTokenIDs(models: models, task: task, audioTokenCount: audioTokenCount)
        var promptEmbeds = try runTokenEmbedding(model: models.tokenEmbeddingModel, tokenIDs: tokenIDs)
        try scatterAudioEmbeds(
            audioEmbeds,
            into: &promptEmbeds,
            tokenIDs: tokenIDs,
            audioTokenID: models.manifest.audioTokenID,
            audioTokenCount: audioTokenCount
        )
        return GranitePlusPreparedPrompt(embeddings: promptEmbeds, tokenCount: tokenIDs.count)
    }

    private func generate(
        logits: inout MLMultiArray,
        state: MLState,
        startPosition: Int,
        maxNewTokens: Int,
        models: GranitePlusAsrModels
    ) async throws -> (tokenIDs: [Int], stoppedOnEOS: Bool) {
        var generated: [Int] = []
        generated.reserveCapacity(min(maxNewTokens, models.manifest.maxSequenceLength - startPosition))
        var position = startPosition
        let tokenBudget = min(maxNewTokens, max(0, models.manifest.maxSequenceLength - position))

        for _ in 0..<tokenBudget {
            let nextID = argmax(logits)
            if nextID == models.manifest.eosTokenID {
                return (generated, true)
            }
            generated.append(nextID)
            let nextEmbeds = try runTokenEmbedding(model: models.tokenEmbeddingModel, tokenIDs: [nextID])
            logits = try await runLanguageModel(
                model: models.languageModel,
                state: state,
                embeddings: nextEmbeds,
                startPosition: position
            )
            position += 1
        }
        return (generated, false)
    }
}

@available(macOS 15, iOS 18, *)
extension GranitePlusAsrManager {
    private func promptTokenIDs(
        models: GranitePlusAsrModels,
        task: GranitePlusTask,
        audioTokenCount: Int
    ) throws -> [Int] {
        let prompt = GranitePlusPromptBuilder.prompt(task: task, audioTokenCount: audioTokenCount)
        let tokenIDs = models.tokenizer.encode(prompt, addSpecialTokens: false)
        guard tokenIDs.filter({ $0 == models.manifest.audioTokenID }).count == audioTokenCount else {
            throw GraniteAsrError.invalidOutput("Granite Plus prompt/audio token count mismatch")
        }
        guard tokenIDs.count <= models.manifest.maxQueryLength else {
            throw GraniteAsrError.invalidOutput("Granite Plus prompt has \(tokenIDs.count) tokens")
        }
        return tokenIDs
    }

    private func runAudioModel(
        models: GranitePlusAsrModels,
        meta: GranitePlusAudioWindowMeta,
        features: MLMultiArray
    ) throws -> MLMultiArray {
        let inputName = meta.inputs.first ?? "input_features"
        let outputName = meta.outputs.first ?? "audio_embeds"
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: features)
        ])
        let output = try models.audioModel.prediction(from: provider)
        guard let audioEmbeds = output.featureValue(for: outputName)?.multiArrayValue else {
            throw GraniteAsrError.invalidOutput("Missing \(outputName)")
        }
        return audioEmbeds
    }

    private func runTokenEmbedding(model: MLModel, tokenIDs: [Int]) throws -> MLMultiArray {
        let inputIDs = try makeIntArray(shape: [1, tokenIDs.count], values: tokenIDs.map(Int32.init))
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs)
        ])
        let output = try model.prediction(from: provider)
        guard let embeddings = output.featureValue(for: "inputs_embeds")?.multiArrayValue else {
            throw GraniteAsrError.invalidOutput("Missing inputs_embeds")
        }
        return embeddings
    }

    private func runLanguageModel(
        model: MLModel,
        state: MLState,
        embeddings: MLMultiArray,
        startPosition: Int
    ) async throws -> MLMultiArray {
        let queryLength = embeddings.shape[1].intValue
        let positions = try positionIDs(start: startPosition, count: queryLength)
        let mask = try causalMask(queryLength: queryLength, pastLength: startPosition)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "inputs_embeds": MLFeatureValue(multiArray: embeddings),
            "position_ids": MLFeatureValue(multiArray: positions),
            "causal_mask": MLFeatureValue(multiArray: mask),
        ])
        let output = try await model.prediction(from: provider, using: state)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw GraniteAsrError.invalidOutput("Missing logits")
        }
        return logits
    }

    private func scatterAudioEmbeds(
        _ audioEmbeds: MLMultiArray,
        into promptEmbeds: inout MLMultiArray,
        tokenIDs: [Int],
        audioTokenID: Int,
        audioTokenCount: Int
    ) throws {
        let promptPtr = promptEmbeds.dataPointer.bindMemory(to: Float.self, capacity: promptEmbeds.count)
        let audioPtr = audioEmbeds.dataPointer.bindMemory(to: Float.self, capacity: audioEmbeds.count)
        let hiddenSize = promptEmbeds.shape[2].intValue
        var audioIndex = 0

        for (tokenIndex, tokenID) in tokenIDs.enumerated() where tokenID == audioTokenID {
            guard audioIndex < audioTokenCount else {
                throw GraniteAsrError.invalidOutput("Granite Plus prompt has too many audio tokens")
            }
            let promptOffset = tokenIndex * hiddenSize
            let audioOffset = audioIndex * hiddenSize
            guard audioOffset + hiddenSize <= audioEmbeds.count else {
                throw GraniteAsrError.invalidOutput("Granite Plus audio embedding overflow")
            }
            promptPtr
                .advanced(by: promptOffset)
                .update(from: audioPtr.advanced(by: audioOffset), count: hiddenSize)
            audioIndex += 1
        }
        guard audioIndex == audioTokenCount else {
            throw GraniteAsrError.invalidOutput("Granite Plus audio token count mismatch")
        }
    }

    private func validAudioTokenCount(validEncoderFrames: Int, meta: GranitePlusAudioWindowMeta) -> Int {
        let qFormerWindow = 15
        let qFormerQueries = 3
        let count = ((validEncoderFrames + qFormerWindow - 1) / qFormerWindow) * qFormerQueries
        return max(1, min(meta.audioTokens, count))
    }

    private func argmax(_ logits: MLMultiArray) -> Int {
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
        var bestIndex = 0
        var bestValue = ptr[0]
        for index in 1..<logits.count where ptr[index] > bestValue {
            bestIndex = index
            bestValue = ptr[index]
        }
        return bestIndex
    }

    private func positionIDs(start: Int, count: Int) throws -> MLMultiArray {
        let cacheKey = start * 4096 + count
        if let cached = positionCache[cacheKey] {
            return cached
        }
        let values = (start..<start + count).map(Int32.init)
        let array = try makeIntArray(shape: [1, count], values: values)
        positionCache[cacheKey] = array
        return array
    }

    private func causalMask(queryLength: Int, pastLength: Int) throws -> MLMultiArray {
        let key = "\(queryLength):\(pastLength)"
        if let cached = causalMaskCache[key] {
            return cached
        }
        let totalLength = queryLength + pastLength
        let array = try MLMultiArray(
            shape: [1, 1, NSNumber(value: queryLength), NSNumber(value: totalLength)],
            dataType: .float32
        )
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        for queryIndex in 0..<queryLength {
            let rowOffset = queryIndex * totalLength
            let validUntil = pastLength + queryIndex
            for keyIndex in 0..<totalLength {
                ptr[rowOffset + keyIndex] = keyIndex <= validUntil ? 0 : -10_000
            }
        }
        causalMaskCache[key] = array
        return array
    }

    private func makeIntArray(shape: [Int], values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        for index in 0..<values.count {
            ptr[index] = values[index]
        }
        return array
    }
}

@available(macOS 15, iOS 18, *)
private struct GranitePlusChunkWindow: Sendable {
    let index: Int
    let startSample: Int
    let endSample: Int
    let windowSamples: Int
    let sampleRate: Int

    var startSeconds: Double {
        Double(startSample) / Double(sampleRate)
    }
}

@available(macOS 15, iOS 18, *)
private func makePlusChunks(
    totalSamples: Int,
    sampleRate: Int,
    windowSeconds: Int,
    overlapSeconds: Double,
    minTailSeconds: Double
) throws -> [GranitePlusChunkWindow] {
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

    var chunks: [GranitePlusChunkWindow] = []
    var start = 0
    while start < totalSamples {
        let end = min(start + windowSamples, totalSamples)
        chunks.append(
            GranitePlusChunkWindow(
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

@available(macOS 15, iOS 18, *)
enum GranitePlusChunkMerge {
    private struct Token {
        let original: String
        let normalized: String
    }

    static func mergeTranscriptChunks(_ parts: [String]) -> String {
        guard let first = parts.first else { return "" }

        var accumulator = first.trimmingCharacters(in: .whitespacesAndNewlines)
        for next in parts.dropFirst() {
            accumulator = self.mergeTwo(accumulator, next)
        }
        return accumulator.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func mergeTwo(_ lhs: String, _ rhs: String) -> String {
        let lhs = lhs.trimmingCharacters(in: .whitespacesAndNewlines)
        let rhs = rhs.trimmingCharacters(in: .whitespacesAndNewlines)
        let lhsTokens = self.tokenize(lhs)
        let rhsTokens = self.tokenize(rhs)

        if lhsTokens.isEmpty { return rhs }
        if rhsTokens.isEmpty { return lhs }

        let maxOverlap = min(lhsTokens.count, rhsTokens.count, 80)
        if let overlap = self.bestOverlap(lhsTokens: lhsTokens, rhsTokens: rhsTokens, maxOverlap: maxOverlap) {
            let remainder = rhsTokens.dropFirst(overlap).map(\.original).joined(separator: " ")
            return remainder.isEmpty ? lhs : lhs + " " + remainder
        }

        return lhs + " " + rhs
    }

    private static func bestOverlap(lhsTokens: [Token], rhsTokens: [Token], maxOverlap: Int) -> Int? {
        guard maxOverlap > 0 else { return nil }

        for overlap in stride(from: maxOverlap, through: 2, by: -1) {
            if self.exactNormalizedMatch(lhsTokens: lhsTokens, rhsTokens: rhsTokens, overlap: overlap) {
                return overlap
            }
        }

        for overlap in stride(from: maxOverlap, through: 4, by: -1) {
            if self.fuzzyNormalizedMatch(lhsTokens: lhsTokens, rhsTokens: rhsTokens, overlap: overlap) {
                return overlap
            }
        }

        return nil
    }

    private static func exactNormalizedMatch(lhsTokens: [Token], rhsTokens: [Token], overlap: Int) -> Bool {
        zip(lhsTokens.suffix(overlap), rhsTokens.prefix(overlap)).allSatisfy { lhs, rhs in
            lhs.normalized.isEmpty == false && lhs.normalized == rhs.normalized
        }
    }

    private static func fuzzyNormalizedMatch(lhsTokens: [Token], rhsTokens: [Token], overlap: Int) -> Bool {
        var matches = 0
        var informativeMatches = 0

        for (lhs, rhs) in zip(lhsTokens.suffix(overlap), rhsTokens.prefix(overlap)) {
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

        let requiredMatches = max(4, Int((Double(overlap) * 0.65).rounded(.up)))
        return matches >= requiredMatches && informativeMatches >= 2
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
