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
public actor CohereTranscribeAsrManager {
    private var models: CohereTranscribeAsrModels?
    private let logger = Logger(subsystem: "FluidAudio", category: "CohereTranscribeAsrManager")

    public init() {}

    public func loadModels(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws {
        self.models = try await CohereTranscribeAsrModels.load(from: directory, computeUnits: computeUnits)
        self.logger.info("Cohere Transcribe CoreML models loaded from \(directory.path, privacy: .public)")
    }

    public func transcribe(
        audioSamples: [Float],
        decoderMode: CohereTranscribeDecoderMode = .cached,
        maxNewTokens: Int? = nil
    ) async throws -> String {
        guard let models else {
            throw CohereTranscribeAsrError.generationFailed("Models not loaded")
        }
        guard audioSamples.isEmpty == false else { return "" }

        let overlapSamples = self.effectiveOverlapSamples(for: models.manifest)
        let starts = self.windowStarts(
            totalSamples: audioSamples.count,
            windowSamples: models.manifest.maxAudioSamples,
            overlapSamples: overlapSamples
        )
        let startedAt = Date()
        let audioSeconds = Double(audioSamples.count) / Double(models.manifest.sampleRate)
        self.logger.info(
            "Cohere transcribe start [samples=\(audioSamples.count), audioSeconds=\(audioSeconds, format: .fixed(precision: 2)), decoderMode=\(decoderMode.rawValue, privacy: .public), chunks=\(starts.count), overlapSamples=\(overlapSamples)]"
        )

        var chunkTexts: [String] = []
        chunkTexts.reserveCapacity(starts.count)

        for chunkIndex in 0..<starts.count {
            let frontInputs = try self.buildChunkInputs(
                chunkIndex: chunkIndex,
                starts: starts,
                audioSamples: audioSamples,
                manifest: models.manifest
            )
            let tokenIDs = try self.runPipeline(
                frontInputs: frontInputs,
                models: models,
                decoderMode: decoderMode,
                maxNewTokens: maxNewTokens
            )
            let text = self.decodeTokens(tokenIDs, manifest: models.manifest)
            self.logger.debug(
                "Cohere chunk \(chunkIndex + 1)/\(starts.count) finished [tokenCount=\(tokenIDs.count), charCount=\(text.count)]"
            )
            chunkTexts.append(text)
        }

        let merged = CohereChunkMerge.mergeTranscriptChunks(chunkTexts)
        let elapsed = Date().timeIntervalSince(startedAt)
        let rtf = audioSeconds > 0 ? elapsed / audioSeconds : 0
        self.logger.info(
            "Cohere transcribe finished in \(elapsed, format: .fixed(precision: 2))s [audioSeconds=\(audioSeconds, format: .fixed(precision: 2)), rtf=\(rtf, format: .fixed(precision: 2))x, chars=\(merged.count)]"
        )
        return merged
    }

    public func transcribe(
        audioFileAt url: URL,
        decoderMode: CohereTranscribeDecoderMode = .cached,
        maxNewTokens: Int? = nil
    ) async throws -> String {
        guard let models else {
            throw CohereTranscribeAsrError.generationFailed("Models not loaded")
        }

        let audioConverter = AudioConverter(sampleRate: Double(models.manifest.sampleRate))
        let audioSamples = try audioConverter.resampleAudioFile(url)
        return try await self.transcribe(
            audioSamples: audioSamples,
            decoderMode: decoderMode,
            maxNewTokens: maxNewTokens
        )
    }

    private func runPipeline(
        frontInputs: MLDictionaryFeatureProvider,
        models: CohereTranscribeAsrModels,
        decoderMode: CohereTranscribeDecoderMode,
        maxNewTokens: Int?
    ) throws -> [Int32] {
        switch decoderMode {
        case .fullSequence:
            return try self.runFullSequencePipeline(
                frontInputs: frontInputs,
                models: models,
                maxNewTokens: maxNewTokens
            )
        case .cached:
            guard models.manifest.decoderCached != nil, models.cachedDecoder != nil else {
                self.logger.warning("Cached decoder metadata missing; falling back to full-sequence decode.")
                return try self.runFullSequencePipeline(
                    frontInputs: frontInputs,
                    models: models,
                    maxNewTokens: maxNewTokens
                )
            }
            return try self.runCachedPipeline(
                frontInputs: frontInputs,
                models: models,
                maxNewTokens: maxNewTokens
            )
        }
    }

    private func buildChunkInputs(
        chunkIndex: Int,
        starts: [Int],
        audioSamples: [Float],
        manifest: CohereTranscribeAsrManifest
    ) throws -> MLDictionaryFeatureProvider {
        let start = starts[chunkIndex]
        let end = min(start + manifest.maxAudioSamples, audioSamples.count)
        var segment = Array(audioSamples[start..<end])
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

    private func runFrontendEncoder(
        frontInputs: MLDictionaryFeatureProvider,
        models: CohereTranscribeAsrModels
    ) throws -> (encoderHidden: MLMultiArray, encoderValid: Int) {
        let manifest = models.manifest
        let frontOut = try models.frontend.prediction(from: frontInputs)
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
        let encoderOut = try models.encoder.prediction(from: encoderInputs)
        guard
            let encoderHidden = encoderOut.featureValue(for: manifest.encoder.outputs[0])?.multiArrayValue,
            let encoderLengthOut = encoderOut.featureValue(for: manifest.encoder.outputs[1])?.multiArrayValue
        else {
            throw CohereTranscribeAsrError.invalidOutput("Encoder outputs missing")
        }

        let encoderHiddenContiguous = try self.toContiguous(encoderHidden)
        let encoderValidRaw = Int(encoderLengthOut[0].doubleValue.rounded())
        let encoderValid = max(1, min(manifest.maxEncoderFrames, encoderValidRaw))
        return (encoderHiddenContiguous, encoderValid)
    }

    private func runFullSequencePipeline(
        frontInputs: MLDictionaryFeatureProvider,
        models: CohereTranscribeAsrModels,
        maxNewTokens: Int?
    ) throws -> [Int32] {
        let manifest = models.manifest
        let frontendEncoder = try self.runFrontendEncoder(frontInputs: frontInputs, models: models)
        let encoderHidden = frontendEncoder.encoderHidden
        let encoderValid = frontendEncoder.encoderValid

        let vocabSize = manifest.idToToken.count
        let totalMaxNewTokens = maxNewTokens ?? manifest.defaultMaxNewTokens
        let padToken = Int32(manifest.padTokenID ?? 0)
        var inputIDs = Array(repeating: padToken, count: manifest.decoderMaxLen)
        var attentionMask = Array(repeating: Int32(0), count: manifest.decoderMaxLen)

        for (index, tokenID) in manifest.promptIDs.enumerated() where index < manifest.decoderMaxLen {
            inputIDs[index] = Int32(tokenID)
            attentionMask[index] = 1
        }

        let crossAttentionMask = try self.makeCrossAttentionMask(
            validLength: encoderValid,
            totalLength: manifest.maxEncoderFrames,
            useFloat16: manifest.isFp16
        )

        var currentIndex = manifest.promptIDs.count - 1
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

    private func runCachedPipeline(
        frontInputs: MLDictionaryFeatureProvider,
        models: CohereTranscribeAsrModels,
        maxNewTokens: Int?
    ) throws -> [Int32] {
        guard
            let cachedMetadata = models.manifest.decoderCached,
            let cachedDecoder = models.cachedDecoder
        else {
            throw CohereTranscribeAsrError.generationFailed("Cached decoder is unavailable")
        }

        let manifest = models.manifest
        let frontendEncoder = try self.runFrontendEncoder(frontInputs: frontInputs, models: models)
        let encoderHidden = frontendEncoder.encoderHidden
        let encoderValid = frontendEncoder.encoderValid

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
            validLength: encoderValid,
            totalLength: manifest.maxEncoderFrames,
            useFloat16: manifest.isFp16
        )

        let inputIDArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let stepArray = try MLMultiArray(shape: [1], dataType: .int32)
        let featureProvider = FastFeatureProvider([
            (cachedMetadata.inputs[0], encoderHidden),
            (cachedMetadata.inputs[1], inputIDArray),
            (cachedMetadata.inputs[2], cacheKA),
            (cachedMetadata.inputs[3], cacheVA),
            (cachedMetadata.inputs[4], stepArray),
            (cachedMetadata.inputs[5], crossAttentionMask),
        ])

        let optionsA = MLPredictionOptions()
        optionsA.outputBackings = [
            cachedMetadata.logitsOutput: logitsBuffer,
            cachedMetadata.cacheKOutput: cacheKB,
            cachedMetadata.cacheVOutput: cacheVB,
        ]
        let optionsB = MLPredictionOptions()
        optionsB.outputBackings = [
            cachedMetadata.logitsOutput: logitsBuffer,
            cachedMetadata.cacheKOutput: cacheKA,
            cachedMetadata.cacheVOutput: cacheVA,
        ]

        var useA = true

        func runStep(tokenID: Int32, stepIndex: Int32) throws -> MLMultiArray {
            inputIDArray[0] = NSNumber(value: tokenID)
            stepArray[0] = NSNumber(value: stepIndex)

            let inputCacheK = useA ? cacheKA : cacheKB
            let inputCacheV = useA ? cacheVA : cacheVB
            featureProvider.update(cachedMetadata.inputs[2], inputCacheK)
            featureProvider.update(cachedMetadata.inputs[3], inputCacheV)

            let options = useA ? optionsA : optionsB
            _ = try cachedDecoder.prediction(from: featureProvider, options: options)
            useA.toggle()
            return logitsBuffer
        }

        var generated = manifest.promptIDs.map(Int32.init)
        var lastLogits: MLMultiArray?
        for (index, tokenID) in manifest.promptIDs.enumerated() {
            lastLogits = try runStep(tokenID: Int32(tokenID), stepIndex: Int32(index))
        }

        var currentIndex = manifest.promptIDs.count
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

    private func effectiveOverlapSamples(for manifest: CohereTranscribeAsrManifest) -> Int {
        let maxSamples = manifest.maxAudioSamples
        if maxSamples <= 1 {
            return 0
        }
        if let overlapSamples = manifest.overlapSamples, overlapSamples >= 0 {
            return min(overlapSamples, maxSamples - 1)
        }
        let overlapSeconds = manifest.overlapSeconds ?? 5.0
        let overlap = Int(round(Double(manifest.sampleRate) * overlapSeconds))
        return min(max(0, overlap), maxSamples - 1)
    }

    private func windowStarts(totalSamples: Int, windowSamples: Int, overlapSamples: Int) -> [Int] {
        guard totalSamples > 0, windowSamples > 0 else { return [] }
        guard overlapSamples >= 0, overlapSamples < windowSamples else { return [] }
        if totalSamples <= windowSamples { return [0] }

        let stride = windowSamples - overlapSamples
        var starts: [Int] = []
        var currentStart = 0

        while currentStart + windowSamples < totalSamples {
            starts.append(currentStart)
            currentStart += stride
        }

        let lastStart = totalSamples - windowSamples
        if starts.isEmpty || starts.last != lastStart {
            starts.append(lastStart)
        }
        return starts
    }

    private func decodeTokens(
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

    private func applyPreemphasis(
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

    private func makeCrossAttentionMask(
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

    private func argmax1D(_ array: MLMultiArray) -> Int32 {
        let count = array.count
        if array.dataType == .float16 {
            let pointer = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            var bestIndex = 0
            var bestValue = pointer[0]
            for index in 1..<count {
                let current = pointer[index]
                if Float16(bitPattern: current) > Float16(bitPattern: bestValue) {
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

    private func argmaxSlice(
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

    private func toContiguous(_ array: MLMultiArray) throws -> MLMultiArray {
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

    private func makeFloatArray(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        _ = values.withUnsafeBufferPointer { buffer in
            memcpy(pointer, buffer.baseAddress, values.count * MemoryLayout<Float>.stride)
        }
        return array
    }

    private func makeFloat16Array(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float16)
        let pointer = array.dataPointer.bindMemory(to: UInt16.self, capacity: values.count)
        for index in 0..<values.count {
            var half = Float16(values[index])
            withUnsafeBytes(of: &half) { bytes in
                pointer[index] = bytes.load(as: UInt16.self)
            }
        }
        return array
    }

    private func makeIntArray(shape: [Int], values: [Int32]) throws -> MLMultiArray {
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
