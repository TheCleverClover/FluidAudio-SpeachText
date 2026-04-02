@preconcurrency import AVFoundation
@preconcurrency import CoreML
import Foundation
import OSLog

@available(macOS 15, iOS 18, *)
public enum CohereTranscribeDecoderMode: String, Sendable {
    case cached
    case fullSequence
}

@available(macOS 15, iOS 18, *)
private struct CohereTranscribeEncoderStageOutput: @unchecked Sendable {
    let frontMs: Double
    let encMs: Double
    let crossKVms: Double
    let encoderHidden: MLMultiArray
    let encoderValid: Int
    let crossK: MLMultiArray?
    let crossV: MLMultiArray?
}

@available(macOS 15, iOS 18, *)
public actor CohereTranscribeAsrManager {
    private var models: CohereTranscribeAsrModels?
    private let logger = Logger(subsystem: "FluidAudio", category: "CohereTranscribeAsrManager")

    public init() {}

    public func loadModels(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws {
        try await self.loadModels(
            from: directory,
            computeConfiguration: .init(uniform: computeUnits)
        )
    }

    public func loadModels(
        from directory: URL,
        computeConfiguration: CohereTranscribeComputeConfiguration
    ) async throws {
        self.models = try await CohereTranscribeAsrModels.load(
            from: directory,
            computeConfiguration: computeConfiguration
        )
        self.logger.info("Cohere Transcribe CoreML models loaded from \(directory.path, privacy: .public)")
    }

    public func transcribe(
        audioSamples: [Float],
        decoderMode: CohereTranscribeDecoderMode = .cached,
        maxNewTokens: Int? = nil,
        promptIDs: [Int]? = nil
    ) async throws -> String {
        guard let models else {
            throw CohereTranscribeAsrError.generationFailed("Models not loaded")
        }
        guard audioSamples.isEmpty == false else { return "" }

        let chunks = CohereAudioChunkPlanner.makeChunks(
            audioSamples: audioSamples,
            sampleRate: models.manifest.sampleRate,
            maxAudioSamples: models.manifest.maxAudioSamples,
            boundarySearchSeconds: self.boundarySearchSeconds(for: models.manifest)
        )
        let startedAt = Date()
        let audioSeconds = Double(audioSamples.count) / Double(models.manifest.sampleRate)
        self.logger.info(
            "Cohere transcribe start [samples=\(audioSamples.count), audioSeconds=\(audioSeconds, format: .fixed(precision: 2)), decoderMode=\(decoderMode.rawValue, privacy: .public), chunks=\(chunks.count), chunking=energy-boundary]"
        )

        let chunkTexts: [String]
        if self.shouldUseAsyncOverlap(models: models, decoderMode: decoderMode, chunkCount: chunks.count) {
            self.logger.info("Cohere async overlap enabled for chunked cached decode.")
            chunkTexts = try self.transcribeChunksWithAsyncOverlap(
                chunks: chunks,
                audioSamples: audioSamples,
                models: models,
                maxNewTokens: maxNewTokens,
                promptIDs: promptIDs
            )
        } else {
            var sequentialTexts: [String] = []
            sequentialTexts.reserveCapacity(chunks.count)

            for chunkIndex in 0..<chunks.count {
                let frontInputs = try self.buildChunkInputs(
                    chunk: chunks[chunkIndex],
                    audioSamples: audioSamples,
                    manifest: models.manifest
                )
                let tokenIDs = try self.runPipeline(
                    frontInputs: frontInputs,
                    models: models,
                    decoderMode: decoderMode,
                    maxNewTokens: maxNewTokens,
                    promptIDs: promptIDs
                )
                let text = self.decodeTokens(tokenIDs, manifest: models.manifest)
                self.logger.debug(
                    "Cohere chunk \(chunkIndex + 1)/\(chunks.count) finished [tokenCount=\(tokenIDs.count), charCount=\(text.count), sampleCount=\(chunks[chunkIndex].sampleCount)]"
                )
                sequentialTexts.append(text)
            }

            chunkTexts = sequentialTexts
        }

        let merged = CohereAudioChunkPlanner.joinChunkTexts(chunkTexts)
        let elapsed = Date().timeIntervalSince(startedAt)
        let rtf = audioSeconds > 0 ? elapsed / audioSeconds : 0
        self.logger.info(
            "Cohere transcribe finished in \(elapsed, format: .fixed(precision: 2))s [audioSeconds=\(audioSeconds, format: .fixed(precision: 2)), rtf=\(rtf, format: .fixed(precision: 2))x, chars=\(merged.count)]"
        )
        return merged
    }

    private func shouldUseAsyncOverlap(
        models: CohereTranscribeAsrModels,
        decoderMode: CohereTranscribeDecoderMode,
        chunkCount: Int
    ) -> Bool {
        guard chunkCount > 1, decoderMode == .cached else { return false }
        guard models.computeConfiguration.usesSplitCompute else { return false }
        guard self.canRunCachedDecoder(with: models) else { return false }
        guard models.crossKVProjector != nil, models.manifest.crossKVProjector != nil else { return false }
        return true
    }

    private func transcribeChunksWithAsyncOverlap(
        chunks: [CohereAudioChunkPlanner.Chunk],
        audioSamples: [Float],
        models: CohereTranscribeAsrModels,
        maxNewTokens: Int?,
        promptIDs: [Int]?
    ) throws -> [String] {
        final class PrefetchBox: @unchecked Sendable {
            var result: Result<CohereTranscribeEncoderStageOutput, Error>?
        }

        var texts: [String] = []
        texts.reserveCapacity(chunks.count)

        let queue = DispatchQueue(label: "FluidAudio.CohereAsyncOverlap", qos: .userInitiated)
        let modelsRef = models

        var currentEncoder = try self.runEncoderStage(
            frontInputs: self.buildChunkInputs(
                chunk: chunks[0],
                audioSamples: audioSamples,
                manifest: models.manifest
            ),
            models: models
        )

        for chunkIndex in 0..<chunks.count {
            let nextBox: PrefetchBox?
            let nextGroup: DispatchGroup?

            if chunkIndex + 1 < chunks.count {
                let frontInputs = try self.buildChunkInputs(
                    chunk: chunks[chunkIndex + 1],
                    audioSamples: audioSamples,
                    manifest: models.manifest
                )
                let group = DispatchGroup()
                let box = PrefetchBox()
                group.enter()
                queue.async {
                    defer { group.leave() }
                    box.result = Result {
                        try self.runEncoderStage(frontInputs: frontInputs, models: modelsRef)
                    }
                }
                nextBox = box
                nextGroup = group
            } else {
                nextBox = nil
                nextGroup = nil
            }

            let tokenIDs = try self.runCachedDecoderOnly(
                encoderStage: currentEncoder,
                models: models,
                maxNewTokens: maxNewTokens,
                promptIDs: promptIDs
            )
            let text = self.decodeTokens(tokenIDs, manifest: models.manifest)
            self.logger.debug(
                "Cohere chunk \(chunkIndex + 1)/\(chunks.count) finished [tokenCount=\(tokenIDs.count), charCount=\(text.count), sampleCount=\(chunks[chunkIndex].sampleCount), frontMs=\(currentEncoder.frontMs, format: .fixed(precision: 1)), encMs=\(currentEncoder.encMs, format: .fixed(precision: 1)), crossKVms=\(currentEncoder.crossKVms, format: .fixed(precision: 1))]"
            )
            texts.append(text)

            if let nextGroup, let nextBox {
                nextGroup.wait()
                guard let result = nextBox.result else {
                    throw CohereTranscribeAsrError.generationFailed("Async encoder prefetch produced no result")
                }
                currentEncoder = try result.get()
            }
        }

        return texts
    }

    public func transcribe(
        audioFileAt url: URL,
        decoderMode: CohereTranscribeDecoderMode = .cached,
        maxNewTokens: Int? = nil,
        promptIDs: [Int]? = nil
    ) async throws -> String {
        guard let models else {
            throw CohereTranscribeAsrError.generationFailed("Models not loaded")
        }

        let audioConverter = AudioConverter(sampleRate: Double(models.manifest.sampleRate))
        let audioSamples = try audioConverter.resampleAudioFile(url)
        return try await self.transcribe(
            audioSamples: audioSamples,
            decoderMode: decoderMode,
            maxNewTokens: maxNewTokens,
            promptIDs: promptIDs
        )
    }

    private func runPipeline(
        frontInputs: MLDictionaryFeatureProvider,
        models: CohereTranscribeAsrModels,
        decoderMode: CohereTranscribeDecoderMode,
        maxNewTokens: Int?,
        promptIDs: [Int]?
    ) throws -> [Int32] {
        switch decoderMode {
        case .fullSequence:
            return try self.runFullSequencePipeline(
                frontInputs: frontInputs,
                models: models,
                maxNewTokens: maxNewTokens,
                promptIDs: promptIDs
            )
        case .cached:
            guard self.canRunCachedDecoder(with: models) else {
                self.logger.warning("Cached decoder metadata missing; falling back to full-sequence decode.")
                return try self.runFullSequencePipeline(
                    frontInputs: frontInputs,
                    models: models,
                    maxNewTokens: maxNewTokens,
                    promptIDs: promptIDs
                )
            }
            return try self.runCachedPipeline(
                frontInputs: frontInputs,
                models: models,
                maxNewTokens: maxNewTokens,
                promptIDs: promptIDs
            )
        }
    }

    private func canRunCachedDecoder(with models: CohereTranscribeAsrModels) -> Bool {
        guard let metadata = models.manifest.decoderCached, models.cachedDecoder != nil else {
            return false
        }
        return metadata.cacheKOutput != nil && metadata.cacheVOutput != nil
    }

    nonisolated private func buildChunkInputs(
        chunk: CohereAudioChunkPlanner.Chunk,
        audioSamples: [Float],
        manifest: CohereTranscribeAsrManifest
    ) throws -> MLDictionaryFeatureProvider {
        var segment = Array(audioSamples[chunk.range])
        let rawLength = segment.count

        if segment.count < manifest.maxAudioSamples {
            segment += Array(repeating: 0, count: manifest.maxAudioSamples - segment.count)
        }

        self.applyPreemphasis(
            &segment,
            rawLength: rawLength,
            coefficient: manifest.preemph ?? 0.97
        )

        let audioArray = try self.makeFloatArray(
            shape: [1, manifest.maxAudioSamples],
            values: segment
        )
        let audioLengthArray = try self.makeIntArray(shape: [1], values: [Int32(rawLength)])

        return try MLDictionaryFeatureProvider(dictionary: [
            manifest.frontend.inputs[0]: MLFeatureValue(multiArray: audioArray),
            manifest.frontend.inputs[1]: MLFeatureValue(multiArray: audioLengthArray),
        ])
    }

    nonisolated private func runEncoderStage(
        frontInputs: MLDictionaryFeatureProvider,
        models: CohereTranscribeAsrModels
    ) throws -> CohereTranscribeEncoderStageOutput {
        let manifest = models.manifest
        let frontStartedAt = Date()
        let frontOut = try models.frontend.prediction(from: frontInputs)
        let frontMs = Date().timeIntervalSince(frontStartedAt) * 1000
        guard
            let inputFeatures = frontOut.featureValue(for: manifest.frontend.outputs[0])?.multiArrayValue,
            let featureLength = frontOut.featureValue(for: manifest.frontend.outputs[1])?.multiArrayValue
        else {
            throw CohereTranscribeAsrError.invalidOutput("Frontend outputs missing")
        }

        let encoderInputs = try MLDictionaryFeatureProvider(dictionary: [
            manifest.encoder.inputs[0]: MLFeatureValue(multiArray: inputFeatures),
            manifest.encoder.inputs[1]: MLFeatureValue(multiArray: featureLength),
        ])
        let encoderStartedAt = Date()
        let encoderOut = try models.encoder.prediction(from: encoderInputs)
        let encoderMs = Date().timeIntervalSince(encoderStartedAt) * 1000
        guard
            let encoderHidden = encoderOut.featureValue(for: manifest.encoder.outputs[0])?.multiArrayValue,
            let encoderLengthOut = encoderOut.featureValue(for: manifest.encoder.outputs[1])?.multiArrayValue
        else {
            throw CohereTranscribeAsrError.invalidOutput("Encoder outputs missing")
        }

        let encoderHiddenContiguous = try self.toContiguous(encoderHidden)
        let encoderValidRaw = Int(encoderLengthOut[0].doubleValue.rounded())
        let encoderValid = max(1, min(manifest.maxEncoderFrames, encoderValidRaw))

        var crossK: MLMultiArray?
        var crossV: MLMultiArray?
        var crossKVms: Double = 0
        if
            let crossKVMeta = manifest.crossKVProjector,
            let crossKVProjector = models.crossKVProjector
        {
            let inputName = crossKVMeta.inputs.first ?? "encoder_hidden_states"
            let crossKVStartedAt = Date()
            let crossKVInputs = try MLDictionaryFeatureProvider(dictionary: [
                inputName: MLFeatureValue(multiArray: encoderHiddenContiguous)
            ])
            let crossKVOutput = try crossKVProjector.prediction(from: crossKVInputs)
            crossKVms = Date().timeIntervalSince(crossKVStartedAt) * 1000
            if let crossKOutput = crossKVMeta.crossKOutput ?? crossKVMeta.outputs.first {
                crossK = crossKVOutput.featureValue(for: crossKOutput)?.multiArrayValue
            }
            if let crossVOutput = crossKVMeta.crossVOutput ?? crossKVMeta.outputs.dropFirst().first {
                crossV = crossKVOutput.featureValue(for: crossVOutput)?.multiArrayValue
            }
        }

        return CohereTranscribeEncoderStageOutput(
            frontMs: frontMs,
            encMs: encoderMs,
            crossKVms: crossKVms,
            encoderHidden: encoderHiddenContiguous,
            encoderValid: encoderValid,
            crossK: crossK,
            crossV: crossV
        )
    }

    nonisolated private func runFullSequencePipeline(
        frontInputs: MLDictionaryFeatureProvider,
        models: CohereTranscribeAsrModels,
        maxNewTokens: Int?,
        promptIDs: [Int]?
    ) throws -> [Int32] {
        let manifest = models.manifest
        let encoderStage = try self.runEncoderStage(frontInputs: frontInputs, models: models)
        let encoderHidden = encoderStage.encoderHidden
        let encoderValid = encoderStage.encoderValid
        let generationPromptIDs = promptIDs ?? manifest.promptIDs

        let vocabSize = manifest.idToToken.count
        let totalMaxNewTokens = maxNewTokens ?? manifest.defaultMaxNewTokens
        let padToken = Int32(manifest.padTokenID ?? 0)
        var inputIDs = Array(repeating: padToken, count: manifest.decoderMaxLen)
        var attentionMask = Array(repeating: Int32(0), count: manifest.decoderMaxLen)

        for (index, tokenID) in generationPromptIDs.enumerated() where index < manifest.decoderMaxLen {
            inputIDs[index] = Int32(tokenID)
            attentionMask[index] = 1
        }

        let crossAttentionMask = try self.makeCrossAttentionMask(
            validLength: encoderValid,
            totalLength: manifest.maxEncoderFrames,
            useFloat16: manifest.isFp16
        )

        var currentIndex = generationPromptIDs.count - 1
        for _ in 0..<totalMaxNewTokens {
            if currentIndex + 1 >= manifest.decoderMaxLen {
                break
            }

            let idsArray = try self.makeIntArray(shape: [1, manifest.decoderMaxLen], values: inputIDs)
            let maskArray = try self.makeIntArray(shape: [1, manifest.decoderMaxLen], values: attentionMask)
            let decoderInputs = try MLDictionaryFeatureProvider(dictionary: [
                manifest.decoder.inputs[0]: MLFeatureValue(multiArray: encoderHidden),
                manifest.decoder.inputs[1]: MLFeatureValue(multiArray: idsArray),
                manifest.decoder.inputs[2]: MLFeatureValue(multiArray: maskArray),
                manifest.decoder.inputs[3]: MLFeatureValue(multiArray: crossAttentionMask),
            ])

            let decoderOut = try models.decoder.prediction(from: decoderInputs)
            guard let logits = decoderOut.featureValue(for: manifest.decoder.outputs[0])?.multiArrayValue else {
                throw CohereTranscribeAsrError.invalidOutput("Decoder logits missing")
            }

            let next = self.argmaxSlice(logits, tokenIndex: currentIndex, vocabSize: vocabSize)
            currentIndex += 1
            inputIDs[currentIndex] = next
            attentionMask[currentIndex] = 1

            if let eosTokenID = manifest.eosTokenID, Int(next) == eosTokenID {
                break
            }
        }

        return Array(inputIDs[0...currentIndex])
    }

    nonisolated private func runCachedPipeline(
        frontInputs: MLDictionaryFeatureProvider,
        models: CohereTranscribeAsrModels,
        maxNewTokens: Int?,
        promptIDs: [Int]?
    ) throws -> [Int32] {
        let encoderStage = try self.runEncoderStage(frontInputs: frontInputs, models: models)
        return try self.runCachedDecoderOnly(
            encoderStage: encoderStage,
            models: models,
            maxNewTokens: maxNewTokens,
            promptIDs: promptIDs
        )
    }

    nonisolated private func runCachedDecoderOnly(
        encoderStage: CohereTranscribeEncoderStageOutput,
        models: CohereTranscribeAsrModels,
        maxNewTokens: Int?,
        promptIDs: [Int]?
    ) throws -> [Int32] {
        guard
            let cachedMetadata = models.manifest.decoderCached,
            let cachedDecoder = models.cachedDecoder,
            let cacheKOutput = cachedMetadata.cacheKOutput,
            let cacheVOutput = cachedMetadata.cacheVOutput
        else {
            throw CohereTranscribeAsrError.generationFailed("Cached decoder is unavailable")
        }

        let manifest = models.manifest
        let generationPromptIDs = promptIDs ?? manifest.promptIDs
        let maxTokens = maxNewTokens ?? manifest.defaultMaxNewTokens
        let cacheShape = [cachedMetadata.numLayers, cachedMetadata.numHeads, manifest.decoderMaxLen, cachedMetadata.headDim]
            .map(NSNumber.init)
        let cacheType: MLMultiArrayDataType = manifest.isFp16 ? .float16 : .float32
        let cacheBytesPerElement = manifest.isFp16 ? 2 : 4
        let cacheElementCount = cachedMetadata.numLayers * cachedMetadata.numHeads * manifest.decoderMaxLen * cachedMetadata.headDim

        let vocabSize = manifest.idToToken.count
        let logitsShape = [1, vocabSize].map(NSNumber.init)
        let logitsBuffer = try MLMultiArray(shape: logitsShape, dataType: cacheType)

        let cacheKA = try MLMultiArray(shape: cacheShape, dataType: cacheType)
        let cacheVA = try MLMultiArray(shape: cacheShape, dataType: cacheType)
        let cacheKB = try MLMultiArray(shape: cacheShape, dataType: cacheType)
        let cacheVB = try MLMultiArray(shape: cacheShape, dataType: cacheType)
        memset(cacheKA.dataPointer, 0, cacheElementCount * cacheBytesPerElement)
        memset(cacheVA.dataPointer, 0, cacheElementCount * cacheBytesPerElement)

        let crossAttentionMask = try self.makeCrossAttentionMask(
            validLength: encoderStage.encoderValid,
            totalLength: manifest.maxEncoderFrames,
            useFloat16: manifest.isFp16
        )

        let inputIDArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let stepArray = try MLMultiArray(shape: [1], dataType: .int32)
        let inputIDName = self.cachedDecoderInputName(metadata: cachedMetadata, preferred: "input_id", fallbackIndex: 1)
        let cacheKInputName = self.cachedDecoderInputName(metadata: cachedMetadata, preferred: "cache_k", fallbackIndex: 2)
        let cacheVInputName = self.cachedDecoderInputName(metadata: cachedMetadata, preferred: "cache_v", fallbackIndex: 3)
        let stepInputName = self.cachedDecoderInputName(metadata: cachedMetadata, preferred: "step", fallbackIndex: 4)
        let crossAttentionMaskName = self.cachedDecoderInputName(metadata: cachedMetadata, preferred: "cross_attention_mask", fallbackIndex: 5)

        let featureProvider = self.makeCachedDecoderFeatureProvider(
            metadata: cachedMetadata,
            encoderStage: encoderStage,
            inputIDArray: inputIDArray,
            cacheKA: cacheKA,
            cacheVA: cacheVA,
            stepArray: stepArray,
            crossAttentionMask: crossAttentionMask
        )

        let optionsA = MLPredictionOptions()
        optionsA.outputBackings = [
            cachedMetadata.logitsOutput: logitsBuffer,
            cacheKOutput: cacheKB,
            cacheVOutput: cacheVB,
        ]
        let optionsB = MLPredictionOptions()
        optionsB.outputBackings = [
            cachedMetadata.logitsOutput: logitsBuffer,
            cacheKOutput: cacheKA,
            cacheVOutput: cacheVA,
        ]

        var useA = true

        func runStep(tokenID: Int32, stepIndex: Int32) throws -> MLMultiArray {
            inputIDArray[0] = NSNumber(value: tokenID)
            stepArray[0] = NSNumber(value: stepIndex)

            let inputCacheK = useA ? cacheKA : cacheKB
            let inputCacheV = useA ? cacheVA : cacheVB
            featureProvider.update(inputIDName, inputIDArray)
            featureProvider.update(cacheKInputName, inputCacheK)
            featureProvider.update(cacheVInputName, inputCacheV)
            featureProvider.update(stepInputName, stepArray)
            featureProvider.update(crossAttentionMaskName, crossAttentionMask)

            let options = useA ? optionsA : optionsB
            _ = try cachedDecoder.prediction(from: featureProvider, options: options)
            useA.toggle()
            return logitsBuffer
        }

        var generated = generationPromptIDs.map(Int32.init)
        var lastLogits: MLMultiArray?
        for (index, tokenID) in generationPromptIDs.enumerated() {
            lastLogits = try runStep(tokenID: Int32(tokenID), stepIndex: Int32(index))
        }

        var currentIndex = generationPromptIDs.count
        if let lastLogits {
            let next = self.argmax1D(lastLogits)
            generated.append(next)
            if let eosTokenID = manifest.eosTokenID, Int(next) == eosTokenID {
                return generated
            }
        }

        for _ in 1..<maxTokens {
            if currentIndex >= manifest.decoderMaxLen {
                break
            }
            let logits = try runStep(tokenID: generated.last ?? 0, stepIndex: Int32(currentIndex))
            let next = self.argmax1D(logits)
            generated.append(next)
            currentIndex += 1
            if let eosTokenID = manifest.eosTokenID, Int(next) == eosTokenID {
                break
            }
        }

        return generated
    }

    nonisolated private func makeCachedDecoderFeatureProvider(
        metadata: CohereTranscribeCachedDecoderMeta,
        encoderStage: CohereTranscribeEncoderStageOutput,
        inputIDArray: MLMultiArray,
        cacheKA: MLMultiArray,
        cacheVA: MLMultiArray,
        stepArray: MLMultiArray,
        crossAttentionMask: MLMultiArray
    ) -> FastFeatureProvider {
        let inputIDName = self.cachedDecoderInputName(metadata: metadata, preferred: "input_id", fallbackIndex: 1)
        let cacheKInputName = self.cachedDecoderInputName(metadata: metadata, preferred: "cache_k", fallbackIndex: 2)
        let cacheVInputName = self.cachedDecoderInputName(metadata: metadata, preferred: "cache_v", fallbackIndex: 3)
        let stepInputName = self.cachedDecoderInputName(metadata: metadata, preferred: "step", fallbackIndex: 4)
        let crossAttentionMaskName = self.cachedDecoderInputName(metadata: metadata, preferred: "cross_attention_mask", fallbackIndex: 5)

        if let crossK = encoderStage.crossK, let crossV = encoderStage.crossV {
            let crossKName = self.cachedDecoderInputName(metadata: metadata, preferred: "cross_k", fallbackIndex: 0)
            let crossVName = self.cachedDecoderInputName(metadata: metadata, preferred: "cross_v", fallbackIndex: 1)
            return FastFeatureProvider([
                (crossKName, crossK),
                (crossVName, crossV),
                (inputIDName, inputIDArray),
                (cacheKInputName, cacheKA),
                (cacheVInputName, cacheVA),
                (stepInputName, stepArray),
                (crossAttentionMaskName, crossAttentionMask),
            ])
        }

        let encoderHiddenName = self.cachedDecoderInputName(metadata: metadata, preferred: "encoder_hidden_states", fallbackIndex: 0)
        return FastFeatureProvider([
            (encoderHiddenName, encoderStage.encoderHidden),
            (inputIDName, inputIDArray),
            (cacheKInputName, cacheKA),
            (cacheVInputName, cacheVA),
            (stepInputName, stepArray),
            (crossAttentionMaskName, crossAttentionMask),
        ])
    }

    nonisolated private func cachedDecoderInputName(
        metadata: CohereTranscribeCachedDecoderMeta,
        preferred: String,
        fallbackIndex: Int
    ) -> String {
        if metadata.inputs.contains(preferred) {
            return preferred
        }
        if metadata.inputs.indices.contains(fallbackIndex) {
            return metadata.inputs[fallbackIndex]
        }
        return preferred
    }

    nonisolated private func decodeTokens(
        _ tokenIDs: [Int32],
        manifest: CohereTranscribeAsrManifest
    ) -> String {
        var pieces: [String] = []
        pieces.reserveCapacity(tokenIDs.count)

        for tokenID in tokenIDs {
            let tokenIndex = Int(tokenID)
            if let eosTokenID = manifest.eosTokenID, tokenIndex == eosTokenID {
                break
            }
            if let padTokenID = manifest.padTokenID, tokenIndex == padTokenID {
                continue
            }
            guard manifest.idToToken.indices.contains(tokenIndex) else { continue }

            let piece = manifest.idToToken[tokenIndex]
            if piece.hasPrefix("<"), piece.hasSuffix(">") {
                continue
            }
            pieces.append(piece)
        }

        return pieces.joined()
            .replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    nonisolated private func boundarySearchSeconds(for manifest: CohereTranscribeAsrManifest) -> Double {
        if let overlapSamples = manifest.overlapSamples, overlapSamples > 0 {
            return Double(overlapSamples) / Double(manifest.sampleRate)
        }
        if let overlapSeconds = manifest.overlapSeconds, overlapSeconds > 0 {
            return overlapSeconds
        }
        return CohereAudioChunkPlanner.defaultBoundarySearchSeconds
    }

    nonisolated private func applyPreemphasis(
        _ audio: inout [Float],
        rawLength: Int,
        coefficient: Float
    ) {
        let count = min(rawLength, audio.count)
        guard count > 0 else { return }

        if count > 1 {
            for index in stride(from: count - 1, through: 1, by: -1) {
                audio[index] = audio[index] - coefficient * audio[index - 1]
            }
        }

        for index in count..<audio.count {
            audio[index] = 0
        }
    }

    nonisolated private func makeCrossAttentionMask(
        validLength: Int,
        totalLength: Int,
        useFloat16: Bool
    ) throws -> MLMultiArray {
        let limitedValidLength = max(0, min(validLength, totalLength))
        let invalidValue: Float = -1e9
        var values = Array(repeating: invalidValue, count: totalLength)
        if limitedValidLength > 0 {
            for index in 0..<limitedValidLength {
                values[index] = 0
            }
        }

        if useFloat16 {
            return try self.makeFloat16Array(shape: [1, 1, 1, totalLength], values: values)
        }
        return try self.makeFloatArray(shape: [1, 1, 1, totalLength], values: values)
    }

    nonisolated private func argmax1D(_ array: MLMultiArray) -> Int32 {
        let count = array.count
        if array.dataType == .float16 {
            let pointer = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            var bestIndex = 0
            var bestValue = pointer[0]
            for index in 1..<count {
                let current = pointer[index]
                if self.float32(fromFloat16BitPattern: current) > self.float32(fromFloat16BitPattern: bestValue) {
                    bestValue = current
                    bestIndex = index
                }
            }
            return Int32(bestIndex)
        }

        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        var bestIndex = 0
        var bestValue = pointer[0]
        for index in 1..<count {
            if pointer[index] > bestValue {
                bestValue = pointer[index]
                bestIndex = index
            }
        }
        return Int32(bestIndex)
    }

    nonisolated private func argmaxSlice(
        _ array: MLMultiArray,
        tokenIndex: Int,
        vocabSize: Int
    ) -> Int32 {
        let strideT = array.strides[1].intValue
        let strideV = array.strides[2].intValue
        let base = tokenIndex * strideT
        var bestValue = array[base].doubleValue
        var bestIndex = 0

        for index in 1..<vocabSize {
            let value = array[base + index * strideV].doubleValue
            if value > bestValue {
                bestValue = value
                bestIndex = index
            }
        }

        return Int32(bestIndex)
    }

    nonisolated private func toContiguous(_ array: MLMultiArray) throws -> MLMultiArray {
        let shape = array.shape.map(\.intValue)
        let totalCount = shape.reduce(1, *)
        let strides = array.strides.map(\.intValue)

        var expectedStride = 1
        var isContiguous = true
        for index in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[index] != expectedStride {
                isContiguous = false
                break
            }
            expectedStride *= shape[index]
        }

        if isContiguous {
            return array
        }

        let contiguous = try MLMultiArray(shape: array.shape, dataType: array.dataType)
        if array.dataType == .float16 {
            let source = array.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)
            let destination = contiguous.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)
            if shape.count == 3 {
                var outputIndex = 0
                for batch in 0..<shape[0] {
                    for time in 0..<shape[1] {
                        let base = batch * strides[0] + time * strides[1]
                        if strides[2] == 1 {
                            memcpy(destination + outputIndex, source + base, shape[2] * MemoryLayout<UInt16>.size)
                            outputIndex += shape[2]
                        } else {
                            for hidden in 0..<shape[2] {
                                destination[outputIndex] = source[base + hidden * strides[2]]
                                outputIndex += 1
                            }
                        }
                    }
                }
            }
        } else {
            let source = array.dataPointer.bindMemory(to: Float.self, capacity: totalCount)
            let destination = contiguous.dataPointer.bindMemory(to: Float.self, capacity: totalCount)
            if shape.count == 3 {
                var outputIndex = 0
                for batch in 0..<shape[0] {
                    for time in 0..<shape[1] {
                        let base = batch * strides[0] + time * strides[1]
                        if strides[2] == 1 {
                            memcpy(destination + outputIndex, source + base, shape[2] * MemoryLayout<Float>.size)
                            outputIndex += shape[2]
                        } else {
                            for hidden in 0..<shape[2] {
                                destination[outputIndex] = source[base + hidden * strides[2]]
                                outputIndex += 1
                            }
                        }
                    }
                }
            }
        }
        return contiguous
    }

    nonisolated private func makeFloatArray(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        _ = values.withUnsafeBufferPointer { buffer in
            memcpy(pointer, buffer.baseAddress, values.count * MemoryLayout<Float>.stride)
        }
        return array
    }

    nonisolated private func makeFloat16Array(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float16)
        let pointer = array.dataPointer.bindMemory(to: UInt16.self, capacity: values.count)
        for index in 0..<values.count {
            pointer[index] = self.float16BitPattern(from: values[index])
        }
        return array
    }

    nonisolated private func float32(fromFloat16BitPattern bitPattern: UInt16) -> Float {
        let sign = UInt32(bitPattern & 0x8000) << 16
        let exponent = Int((bitPattern >> 10) & 0x1F)
        let fraction = UInt32(bitPattern & 0x03FF)

        let floatBits: UInt32
        switch exponent {
        case 0:
            if fraction == 0 {
                floatBits = sign
            } else {
                var mantissa = fraction
                var adjustedExponent = -14
                while (mantissa & 0x0400) == 0 {
                    mantissa <<= 1
                    adjustedExponent -= 1
                }
                mantissa &= 0x03FF
                let exponentBits = UInt32(adjustedExponent + 127) << 23
                let mantissaBits = mantissa << 13
                floatBits = sign | exponentBits | mantissaBits
            }
        case 0x1F:
            floatBits = sign | 0x7F80_0000 | (fraction << 13)
        default:
            let exponentBits = UInt32(exponent - 15 + 127) << 23
            let mantissaBits = fraction << 13
            floatBits = sign | exponentBits | mantissaBits
        }

        return Float(bitPattern: floatBits)
    }

    nonisolated private func float16BitPattern(from value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        var exponent = Int((bits >> 23) & 0xFF) - 127 + 15
        var mantissa = bits & 0x007F_FFFF

        if exponent <= 0 {
            if exponent < -10 {
                return sign
            }
            mantissa |= 0x0080_0000
            let shift = UInt32(14 - exponent)
            let rounded = (mantissa >> shift) + ((mantissa >> (shift - 1)) & 1)
            return sign | UInt16(rounded)
        }

        if exponent >= 0x1F {
            return sign | 0x7C00
        }

        mantissa = mantissa + 0x0000_1000
        if (mantissa & 0x0080_0000) != 0 {
            mantissa = 0
            exponent += 1
            if exponent >= 0x1F {
                return sign | 0x7C00
            }
        }

        return sign | UInt16(exponent << 10) | UInt16(mantissa >> 13)
    }

    nonisolated private func makeIntArray(shape: [Int], values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .int32)
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        _ = values.withUnsafeBufferPointer { buffer in
            memcpy(pointer, buffer.baseAddress, values.count * MemoryLayout<Int32>.stride)
        }
        return array
    }
}

@available(macOS 15, iOS 18, *)
private final class FastFeatureProvider: MLFeatureProvider {
    private var values: [String: MLFeatureValue]
    let featureNames: Set<String>

    init(_ pairs: [(String, MLMultiArray)]) {
        var values = [String: MLFeatureValue]()
        values.reserveCapacity(pairs.count)
        var names = Set<String>()
        names.reserveCapacity(pairs.count)

        for (name, array) in pairs {
            values[name] = MLFeatureValue(multiArray: array)
            names.insert(name)
        }

        self.values = values
        self.featureNames = names
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        self.values[featureName]
    }

    func update(_ name: String, _ array: MLMultiArray) {
        self.values[name] = MLFeatureValue(multiArray: array)
    }
}
