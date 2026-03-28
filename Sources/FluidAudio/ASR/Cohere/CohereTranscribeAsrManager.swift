@preconcurrency import CoreML
import Foundation
import OSLog

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
        maxNewTokens: Int? = nil
    ) async throws -> String {
        guard let models else {
            throw CohereTranscribeAsrError.generationFailed("Models not loaded")
        }

        let segments = self.makeSegments(
            audioSamples,
            maxSamples: models.manifest.maxAudioSamples,
            overlapSamples: models.manifest.overlapSamples
        )

        var merged = ""
        for segment in segments {
            let text = try self.transcribeChunk(segment, with: models, maxNewTokens: maxNewTokens)
            merged = self.merge(existing: merged, next: text)
        }

        return merged.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func transcribeChunk(
        _ samples: ArraySlice<Float>,
        with models: CohereTranscribeAsrModels,
        maxNewTokens: Int?
    ) throws -> String {
        let manifest = models.manifest
        let actualLength = min(samples.count, manifest.maxAudioSamples)
        guard actualLength > 0 else { return "" }

        var padded = [Float](repeating: 0, count: manifest.maxAudioSamples)
        padded.replaceSubrange(0..<actualLength, with: samples.prefix(actualLength))

        let frontendInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_samples": MLFeatureValue(
                multiArray: try self.makeFloat32Array(
                    shape: [1, manifest.maxAudioSamples],
                    values: padded
                )
            ),
            "audio_length": MLFeatureValue(
                multiArray: try self.makeInt32Array(shape: [1], values: [Int32(actualLength)])
            ),
        ])
        let frontendOutput = try models.frontend.prediction(from: frontendInput)

        guard
            let inputFeatures = frontendOutput.featureValue(for: "var_6916")?.multiArrayValue,
            let featureLength = frontendOutput.featureValue(for: "cast_2")?.multiArrayValue
        else {
            throw CohereTranscribeAsrError.invalidOutput("Frontend outputs missing")
        }

        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_features": MLFeatureValue(multiArray: inputFeatures),
            "feature_length": MLFeatureValue(multiArray: featureLength),
        ])
        let encoderOutput = try models.encoder.prediction(from: encoderInput)

        guard
            let encoderHiddenStates = encoderOutput.featureValue(for: "var_8638")?.multiArrayValue,
            let encoderLengthArray = encoderOutput.featureValue(for: "cast_353")?.multiArrayValue
        else {
            throw CohereTranscribeAsrError.invalidOutput("Encoder outputs missing")
        }

        let encoderLength = max(0, min(self.readFirstInt32(from: encoderLengthArray), manifest.maxEncoderFrames))
        let crossAttentionMask = try self.makeCrossAttentionMask(
            validLength: encoderLength,
            totalLength: manifest.maxEncoderFrames
        )

        let promptLength = manifest.promptIDs.count
        guard promptLength > 0, promptLength < manifest.decoderMaxLen else {
            throw CohereTranscribeAsrError.generationFailed("Invalid prompt length \(promptLength)")
        }

        let fullInputIDs = manifest.promptIDs + Array(
            repeating: manifest.padTokenID,
            count: manifest.decoderMaxLen - promptLength
        )
        let fullDecoderMask = Array(repeating: Int32(1), count: promptLength)
            + Array(repeating: Int32(0), count: manifest.decoderMaxLen - promptLength)

        let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_hidden_states": MLFeatureValue(multiArray: encoderHiddenStates),
            "input_ids": MLFeatureValue(
                multiArray: try self.makeInt32Array(
                    shape: [1, manifest.decoderMaxLen],
                    values: fullInputIDs.map(Int32.init)
                )
            ),
            "decoder_attention_mask": MLFeatureValue(
                multiArray: try self.makeInt32Array(
                    shape: [1, manifest.decoderMaxLen],
                    values: fullDecoderMask
                )
            ),
            "cross_attention_mask": MLFeatureValue(multiArray: crossAttentionMask),
        ])

        let decoderOutput = try models.decoder.prediction(from: decoderInput)
        guard let prefillLogits = decoderOutput.featureValue(for: "var_1009")?.multiArrayValue else {
            throw CohereTranscribeAsrError.invalidOutput("Decoder logits missing")
        }

        let vocabSize = manifest.idToToken.count
        let firstToken = self.argmaxFloat16(
            in: prefillLogits,
            offset: (promptLength - 1) * vocabSize,
            count: vocabSize
        )

        var generated: [Int] = []
        if firstToken != manifest.eosTokenID {
            generated.append(firstToken)
        }

        let maxTokens = min(
            maxNewTokens ?? manifest.defaultMaxNewTokens,
            max(0, manifest.decoderMaxLen - promptLength)
        )

        if maxTokens == 0 || firstToken == manifest.eosTokenID {
            return self.decodeTokens(generated, manifest: manifest)
        }

        var cacheK = try self.makeFloat16Array(
            shape: [8, 8, manifest.decoderMaxLen, 128],
            values: Array(repeating: 0, count: 8 * 8 * manifest.decoderMaxLen * 128)
        )
        var cacheV = try self.makeFloat16Array(
            shape: [8, 8, manifest.decoderMaxLen, 128],
            values: Array(repeating: 0, count: 8 * 8 * manifest.decoderMaxLen * 128)
        )
        var step = promptLength
        var currentToken = firstToken

        for _ in 1..<maxTokens {
            let cachedInput = try MLDictionaryFeatureProvider(dictionary: [
                "encoder_hidden_states": MLFeatureValue(multiArray: encoderHiddenStates),
                "input_id": MLFeatureValue(
                    multiArray: try self.makeInt32Array(shape: [1, 1], values: [Int32(currentToken)])
                ),
                "cache_k": MLFeatureValue(multiArray: cacheK),
                "cache_v": MLFeatureValue(multiArray: cacheV),
                "step": MLFeatureValue(
                    multiArray: try self.makeInt32Array(shape: [1], values: [Int32(step)])
                ),
                "cross_attention_mask": MLFeatureValue(multiArray: crossAttentionMask),
            ])

            let cachedOutput = try models.cachedDecoder.prediction(from: cachedInput)
            guard
                let logits = cachedOutput.featureValue(for: "var_2891")?.multiArrayValue,
                let nextCacheK = cachedOutput.featureValue(for: "var_2894")?.multiArrayValue,
                let nextCacheV = cachedOutput.featureValue(for: "var_2897")?.multiArrayValue
            else {
                throw CohereTranscribeAsrError.invalidOutput("Cached decoder outputs missing")
            }

            cacheK = nextCacheK
            cacheV = nextCacheV
            step += 1

            let nextToken = self.argmaxFloat16(in: logits, offset: 0, count: vocabSize)
            if nextToken == manifest.eosTokenID {
                break
            }

            generated.append(nextToken)
            currentToken = nextToken
        }

        return self.decodeTokens(generated, manifest: manifest)
    }

    private func makeSegments(
        _ samples: [Float],
        maxSamples: Int,
        overlapSamples: Int
    ) -> [ArraySlice<Float>] {
        guard samples.count > maxSamples else { return [samples[0..<samples.count]] }

        let stride = max(1, maxSamples - overlapSamples)
        var segments: [ArraySlice<Float>] = []
        var start = 0

        while start < samples.count {
            let end = min(start + maxSamples, samples.count)
            segments.append(samples[start..<end])
            if end >= samples.count { break }
            start += stride
        }

        return segments
    }

    private func merge(existing: String, next: String) -> String {
        let lhs = existing.trimmingCharacters(in: .whitespacesAndNewlines)
        let rhs = next.trimmingCharacters(in: .whitespacesAndNewlines)

        guard lhs.isEmpty == false else { return rhs }
        guard rhs.isEmpty == false else { return lhs }

        let lhsWords = lhs.split(whereSeparator: \.isWhitespace)
        let rhsWords = rhs.split(whereSeparator: \.isWhitespace)
        let maxOverlap = min(12, lhsWords.count, rhsWords.count)

        if maxOverlap > 0 {
            for overlap in stride(from: maxOverlap, through: 1, by: -1) {
                let lhsSuffix = lhsWords.suffix(overlap).map { $0.lowercased() }
                let rhsPrefix = rhsWords.prefix(overlap).map { $0.lowercased() }
                if lhsSuffix == rhsPrefix {
                    let remainder = rhsWords.dropFirst(overlap).joined(separator: " ")
                    if remainder.isEmpty {
                        return lhs
                    }
                    return lhs + " " + remainder
                }
            }
        }

        return lhs + " " + rhs
    }

    private func decodeTokens(
        _ tokenIDs: [Int],
        manifest: CohereTranscribeAsrManifest
    ) -> String {
        guard tokenIDs.isEmpty == false else { return "" }

        var text = ""
        for tokenID in tokenIDs {
            guard manifest.idToToken.indices.contains(tokenID) else { continue }
            let token = manifest.idToToken[tokenID]

            if token == "<pad>" || token == "<unk>" || token.hasPrefix("<|"), token.hasSuffix("|>") {
                continue
            }

            if token.hasPrefix("▁") {
                let piece = String(token.dropFirst())
                if text.isEmpty {
                    text.append(piece)
                } else {
                    text.append(" ")
                    text.append(piece)
                }
                continue
            }

            if token.hasPrefix("Ġ") {
                let piece = String(token.dropFirst())
                if text.isEmpty == false {
                    text.append(" ")
                }
                text.append(piece)
                continue
            }

            text.append(token)
        }

        return text
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func makeCrossAttentionMask(validLength: Int, totalLength: Int) throws -> MLMultiArray {
        let invalidValue = Float16(-10_000)
        var values = [Float16](repeating: invalidValue, count: totalLength)
        if validLength > 0 {
            values.replaceSubrange(0..<min(validLength, totalLength), with: repeatElement(Float16(0), count: min(validLength, totalLength)))
        }
        return try self.makeFloat16Array(shape: [1, 1, 1, totalLength], values: values)
    }

    private func argmaxFloat16(in array: MLMultiArray, offset: Int, count: Int) -> Int {
        let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        var maxValue = Float16.leastNonzeroMagnitude
        var maxIndex = 0

        for index in 0..<count {
            let value = pointer[offset + index]
            if index == 0 || value > maxValue {
                maxValue = value
                maxIndex = index
            }
        }

        return maxIndex
    }

    private func readFirstInt32(from array: MLMultiArray) -> Int {
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: max(1, array.count))
        return Int(pointer[0])
    }

    private func makeFloat32Array(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        _ = values.withUnsafeBufferPointer { buffer in
            memcpy(pointer, buffer.baseAddress, values.count * MemoryLayout<Float>.stride)
        }
        return array
    }

    private func makeFloat16Array(shape: [Int], values: [Float16]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float16)
        let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: values.count)
        _ = values.withUnsafeBufferPointer { buffer in
            memcpy(pointer, buffer.baseAddress, values.count * MemoryLayout<Float16>.stride)
        }
        return array
    }

    private func makeInt32Array(shape: [Int], values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .int32)
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        _ = values.withUnsafeBufferPointer { buffer in
            memcpy(pointer, buffer.baseAddress, values.count * MemoryLayout<Int32>.stride)
        }
        return array
    }
}
