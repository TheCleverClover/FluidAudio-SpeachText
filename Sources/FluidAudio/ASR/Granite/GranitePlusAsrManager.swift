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
public actor GranitePlusAsrManager {
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
        let preparedPrompt = try preparePrompt(
            audioSamples: audioSamples,
            task: task,
            models: models,
            featureExtractor: featureExtractor
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
        let text: String
        if task == .timestamp {
            text = GranitePlusTimestampParser.plainText(from: rawText)
        } else {
            text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - startedAt
        return GranitePlusTranscriptionResult(
            text: text,
            rawText: rawText,
            durationSeconds: Double(audioSamples.count) / Double(models.manifest.sampleRate),
            elapsedSeconds: elapsed,
            generatedTokens: decoded.tokenIDs.count,
            stoppedOnEOS: decoded.stoppedOnEOS,
            task: task
        )
    }

    private func preparePrompt(
        audioSamples: [Float],
        task: GranitePlusTask,
        models: GranitePlusAsrModels,
        featureExtractor: GraniteFeatureExtractor
    ) throws -> GranitePlusPreparedPrompt {
        let manifest = models.manifest
        let windowKey = "\(manifest.defaultWindowSeconds)s"
        guard let windowMeta = manifest.windows[windowKey] else {
            throw GraniteAsrError.modelNotFound("Granite Plus window \(windowKey)")
        }
        let window = try featureExtractor.makeWindow(audio: audioSamples, windowFrames: windowMeta.frames)
        let audioEmbeds = try runAudioModel(models: models, meta: windowMeta, features: window.inputFeatures)
        let tokenIDs = try promptTokenIDs(models: models, task: task, audioTokenCount: windowMeta.audioTokens)
        var promptEmbeds = try runTokenEmbedding(model: models.tokenEmbeddingModel, tokenIDs: tokenIDs)
        try scatterAudioEmbeds(
            audioEmbeds,
            into: &promptEmbeds,
            tokenIDs: tokenIDs,
            audioTokenID: manifest.audioTokenID
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

        for _ in 0 ..< tokenBudget {
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
private extension GranitePlusAsrManager {
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
            "causal_mask": MLFeatureValue(multiArray: mask)
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
        audioTokenID: Int
    ) throws {
        let promptPtr = promptEmbeds.dataPointer.bindMemory(to: Float.self, capacity: promptEmbeds.count)
        let audioPtr = audioEmbeds.dataPointer.bindMemory(to: Float.self, capacity: audioEmbeds.count)
        let hiddenSize = promptEmbeds.shape[2].intValue
        var audioIndex = 0

        for (tokenIndex, tokenID) in tokenIDs.enumerated() where tokenID == audioTokenID {
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
        guard audioIndex * hiddenSize == audioEmbeds.count else {
            throw GraniteAsrError.invalidOutput("Granite Plus unused audio embeddings")
        }
    }

    private func argmax(_ logits: MLMultiArray) -> Int {
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
        var bestIndex = 0
        var bestValue = ptr[0]
        for index in 1 ..< logits.count where ptr[index] > bestValue {
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
        let values = (start ..< start + count).map(Int32.init)
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
        for queryIndex in 0 ..< queryLength {
            let rowOffset = queryIndex * totalLength
            let validUntil = pastLength + queryIndex
            for keyIndex in 0 ..< totalLength {
                ptr[rowOffset + keyIndex] = keyIndex <= validUntil ? 0 : -10_000
            }
        }
        causalMaskCache[key] = array
        return array
    }

    private func makeIntArray(shape: [Int], values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        for index in 0 ..< values.count {
            ptr[index] = values[index]
        }
        return array
    }
}
