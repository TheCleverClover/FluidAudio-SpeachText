import Accelerate
@preconcurrency import CoreML
import Foundation

/// Internal processing pipeline for Nemotron streaming ASR
/// Contains all tensor manipulation and model inference logic
extension NemotronStreamingAsrManager {
    private final class MutableDecoderInput: NSObject, MLFeatureProvider {
        let token: MLMultiArray
        let tokenLength: MLMultiArray
        var hIn: MLMultiArray
        var cIn: MLMultiArray
        let tokenName: String
        let tokenLengthName: String

        init(
            token: MLMultiArray,
            tokenLength: MLMultiArray,
            hIn: MLMultiArray,
            cIn: MLMultiArray,
            tokenName: String,
            tokenLengthName: String
        ) {
            self.token = token
            self.tokenLength = tokenLength
            self.hIn = hIn
            self.cIn = cIn
            self.tokenName = tokenName
            self.tokenLengthName = tokenLengthName
            super.init()
        }

        var featureNames: Set<String> {
            [tokenName, tokenLengthName, "h_in", "c_in"]
        }

        func featureValue(for featureName: String) -> MLFeatureValue? {
            switch featureName {
            case tokenName:
                return MLFeatureValue(multiArray: token)
            case tokenLengthName:
                return MLFeatureValue(multiArray: tokenLength)
            case "h_in":
                return MLFeatureValue(multiArray: hIn)
            case "c_in":
                return MLFeatureValue(multiArray: cIn)
            default:
                return nil
            }
        }
    }

    private final class MutableJointInput: NSObject, MLFeatureProvider {
        let encoderStep: MLMultiArray
        var decoderStep: MLMultiArray
        let encoderName: String
        let decoderName: String

        init(encoderStep: MLMultiArray, decoderStep: MLMultiArray, encoderName: String, decoderName: String) {
            self.encoderStep = encoderStep
            self.decoderStep = decoderStep
            self.encoderName = encoderName
            self.decoderName = decoderName
            super.init()
        }

        var featureNames: Set<String> {
            [encoderName, decoderName]
        }

        func featureValue(for featureName: String) -> MLFeatureValue? {
            switch featureName {
            case encoderName:
                return MLFeatureValue(multiArray: encoderStep)
            case decoderName:
                return MLFeatureValue(multiArray: decoderStep)
            default:
                return nil
            }
        }
    }

    /// Process a single audio chunk through the full pipeline
    internal func processChunk(_ samples: [Float]) async throws {
        if config.modelLayout == .singleEncoder || config.modelLayout == .splitEncoder {
            try await processSingleEncoderChunk(samples)
            return
        }

        guard let preprocessor = preprocessor,
            let encoder = encoder,
            let decoder = decoder,
            let joint = joint,
            let cacheChannel = cacheChannel,
            let cacheTime = cacheTime,
            let cacheLen = cacheLen,
            var currentH = hState,
            var currentC = cState
        else {
            throw ASRError.notInitialized
        }

        // Track decoder state locally to ensure atomicity
        var currentToken = lastToken

        // 1. Preprocessor: audio -> mel spectrogram
        let audioArray = try createAudioArray(samples)
        let audioLen = try MLMultiArray(shape: [1], dataType: .int32)
        audioLen[0] = NSNumber(value: samples.count)

        let preprocInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: audioArray),
            "audio_length": MLFeatureValue(multiArray: audioLen),
        ])

        let preprocOutput = try await preprocessor.prediction(from: preprocInput)
        guard let chunkMel = preprocOutput.featureValue(for: "mel")?.multiArrayValue else {
            throw ASRError.processingFailed("Preprocessor failed to produce mel output")
        }

        // 2. Build encoder input: prepend mel_cache (9 frames) + current chunk mel
        let inputMel = try prependMelCache(to: chunkMel)

        // 3. Encoder with cache
        let melLen = try MLMultiArray(shape: [1], dataType: .int32)
        melLen[0] = NSNumber(value: config.totalMelFrames)

        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: inputMel),
            "mel_length": MLFeatureValue(multiArray: melLen),
            "cache_channel": MLFeatureValue(multiArray: cacheChannel),
            "cache_time": MLFeatureValue(multiArray: cacheTime),
            "cache_len": MLFeatureValue(multiArray: cacheLen),
        ])

        let encoderOutput = try await encoder.prediction(from: encoderInput)

        // Update encoder cache states
        if let newCacheChannel = encoderOutput.featureValue(for: "cache_channel_out")?.multiArrayValue {
            self.cacheChannel = newCacheChannel
        }
        if let newCacheTime = encoderOutput.featureValue(for: "cache_time_out")?.multiArrayValue {
            self.cacheTime = newCacheTime
        }
        if let newCacheLen = encoderOutput.featureValue(for: "cache_len_out")?.multiArrayValue {
            self.cacheLen = newCacheLen
        }

        guard let encoded = encoderOutput.featureValue(for: "encoded")?.multiArrayValue else {
            throw ASRError.processingFailed("Encoder failed to produce output")
        }

        // Save mel cache for next chunk (last 9 frames)
        melCache = try extractMelCache(from: chunkMel)

        // 4. RNNT decode loop for each encoder frame
        let numEncoderFrames = encoded.shape[2].intValue
        var newTokens: [Int] = []

        for t in 0..<numEncoderFrames {
            let encStep = try extractEncoderStep(from: encoded, timeIndex: t)

            // Greedy decode loop (max 10 symbols per frame)
            for _ in 0..<10 {
                let tokenInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
                tokenInput[0] = NSNumber(value: currentToken)

                let tokenLen = try MLMultiArray(shape: [1], dataType: .int32)
                tokenLen[0] = 1

                let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                    "token": MLFeatureValue(multiArray: tokenInput),
                    "token_length": MLFeatureValue(multiArray: tokenLen),
                    "h_in": MLFeatureValue(multiArray: currentH),
                    "c_in": MLFeatureValue(multiArray: currentC),
                ])

                let decoderOutput = try await decoder.prediction(from: decoderInput)

                guard let decoderOut = decoderOutput.featureValue(for: "decoder_out")?.multiArrayValue,
                    let hOut = decoderOutput.featureValue(for: "h_out")?.multiArrayValue,
                    let cOut = decoderOutput.featureValue(for: "c_out")?.multiArrayValue
                else {
                    throw ASRError.processingFailed("Decoder failed")
                }

                // Joint: encoder_step + decoder_out -> logits
                let decoderStep = try sliceDecoderOutput(decoderOut)

                let jointInput = try MLDictionaryFeatureProvider(dictionary: [
                    "encoder": MLFeatureValue(multiArray: encStep),
                    "decoder": MLFeatureValue(multiArray: decoderStep),
                ])

                let jointOutput = try await joint.prediction(from: jointInput)

                guard let logits = jointOutput.featureValue(for: "logits")?.multiArrayValue else {
                    throw ASRError.processingFailed("Joint failed")
                }

                // Find predicted token (index of maximum logit)
                let predToken = findMaxIndex(logits)

                if predToken == config.blankIdx {
                    // Blank token - move to next encoder frame
                    break
                } else {
                    // Non-blank token - emit and update local state
                    newTokens.append(predToken)
                    accumulatedTokenIds.append(predToken)
                    currentToken = Int32(predToken)
                    currentH = hOut
                    currentC = cOut
                }
            }
        }

        // Save final decoder state back to actor properties atomically
        self.lastToken = currentToken
        self.hState = currentH
        self.cState = currentC

        // Invoke partial callback if new tokens were decoded
        if !newTokens.isEmpty, let callback = partialCallback, let tokenizer = tokenizer {
            let partial = tokenizer.decode(ids: accumulatedTokenIds)
            callback(partial)
        }

        processedChunks += 1
    }

    private func processSingleEncoderChunk(_ samples: [Float]) async throws {
        let isFirstChunk = processedChunks == 0
        let activeEncoder: MLModel
        if config.modelLayout == .splitEncoder {
            guard let encoderInit, let encoderStep else {
                throw ASRError.notInitialized
            }
            activeEncoder = isFirstChunk ? encoderInit : encoderStep
        } else {
            guard let encoder else {
                throw ASRError.notInitialized
            }
            activeEncoder = encoder
        }

        guard let preprocessor = preprocessor,
            let decoder = decoder,
            let cacheTime = cacheTime,
            let cacheLen = cacheLen,
            var currentH = hState,
            var currentC = cState
        else {
            throw ASRError.notInitialized
        }
        guard joint != nil || jointDecision != nil else {
            throw ASRError.notInitialized
        }

        let shouldProfile = componentProfilingEnabled
        let chunkInterval = beginProfileInterval("Nemotron.Chunk")
        defer { endProfileInterval("Nemotron.Chunk", chunkInterval) }
        let chunkStartedAt = shouldProfile ? profileNow() : 0
        var currentToken = lastToken
        let audioStartedAt = shouldProfile ? profileNow() : 0
        let audioInterval = beginProfileInterval("Nemotron.AudioInput")
        let audioArray: MLMultiArray
        if let maxAudioSamples = config.maxAudioSamples {
            audioArray = try createPaddedAudioArray(samples, count: maxAudioSamples)
        } else {
            audioArray = try createAudioArray(samples)
        }
        let audioLen = try MLMultiArray(shape: [1], dataType: .int32)
        audioLen[0] = NSNumber(value: samples.count)

        let preprocInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audioArray),
            "audio_length": MLFeatureValue(multiArray: audioLen),
        ])
        if shouldProfile {
            componentProfile.audioInputTime += profileNow() - audioStartedAt
        }
        endProfileInterval("Nemotron.AudioInput", audioInterval)

        let preprocessorStartedAt = shouldProfile ? profileNow() : 0
        let preprocessorInterval = beginProfileInterval("Nemotron.Preprocessor")
        let preprocOutput = try await preprocessor.prediction(from: preprocInput)
        if shouldProfile {
            componentProfile.preprocessorTime += profileNow() - preprocessorStartedAt
        }
        endProfileInterval("Nemotron.Preprocessor", preprocessorInterval)
        guard let processedSignal = preprocOutput.featureValue(for: "processed_signal")?.multiArrayValue else {
            throw ASRError.processingFailed("Preprocessor failed to produce processed_signal output")
        }

        let melStartedAt = shouldProfile ? profileNow() : 0
        let melInterval = beginProfileInterval("Nemotron.MelInput")
        let encoderMelFrames = isFirstChunk ? config.initTotalMelFrames : config.stepTotalMelFrames
        let activePreEncodeCache = isFirstChunk ? config.initPreEncodeCache : config.preEncodeCache
        let encoderMel = try buildSingleEncoderMelInput(
            from: processedSignal,
            totalMelFrames: encoderMelFrames,
            preEncodeCache: activePreEncodeCache
        )
        let melLen = try MLMultiArray(shape: [1], dataType: .int32)
        let validMelFrames = min(processedSignal.shape[2].intValue, encoderMelFrames)
        melLen[0] = NSNumber(value: validMelFrames)

        var encoderFeatures: [String: MLFeatureValue] = [
            "processed_signal": MLFeatureValue(multiArray: encoderMel),
            "processed_signal_length": MLFeatureValue(multiArray: melLen),
            "cache_last_time": MLFeatureValue(multiArray: cacheTime),
            "cache_last_channel_len": MLFeatureValue(multiArray: cacheLen),
        ]
        if !config.encoderStateful {
            guard let cacheChannel = cacheChannel else {
                throw ASRError.notInitialized
            }
            encoderFeatures["cache_last_channel"] = MLFeatureValue(multiArray: cacheChannel)
        }
        if config.runtimePrompt {
            encoderFeatures["prompt_vector"] = MLFeatureValue(multiArray: try createPromptVector())
        }
        let encoderInput = try MLDictionaryFeatureProvider(dictionary: encoderFeatures)
        if shouldProfile {
            componentProfile.melInputTime += profileNow() - melStartedAt
        }
        endProfileInterval("Nemotron.MelInput", melInterval)

        let encoderStartedAt = shouldProfile ? profileNow() : 0
        let encoderInterval = beginProfileInterval("Nemotron.Encoder")
        let encoderOutput: MLFeatureProvider
        if config.encoderStateful {
            guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *) else {
                throw ASRError.processingFailed("Stateful Nemotron encoder requires macOS 15/iOS 18 or newer")
            }
            guard let encoderState = encoderState as? MLState else {
                throw ASRError.notInitialized
            }
            encoderOutput = try await activeEncoder.prediction(from: encoderInput, using: encoderState)
        } else {
            encoderOutput = try await activeEncoder.prediction(from: encoderInput)
        }
        if shouldProfile {
            componentProfile.encoderTime += profileNow() - encoderStartedAt
        }
        endProfileInterval("Nemotron.Encoder", encoderInterval)
        if let newCacheChannel = encoderOutput.featureValue(for: "cache_last_channel_next")?.multiArrayValue {
            self.cacheChannel = newCacheChannel
        }
        if let newCacheTime = encoderOutput.featureValue(for: "cache_last_time_next")?.multiArrayValue {
            self.cacheTime = newCacheTime
        }
        if let newCacheLen = encoderOutput.featureValue(for: "cache_last_channel_next_len")?.multiArrayValue {
            self.cacheLen = newCacheLen
        }

        guard let encoded = encoderOutput.featureValue(for: "encoded")?.multiArrayValue else {
            throw ASRError.processingFailed("Encoder failed to produce encoded output")
        }

        melCache = try extractMelCache(from: processedSignal)
        let numEncoderFrames = validEncoderFrameCount(from: encoderOutput, encoded: encoded)
        try await runGreedyRnntDecodeLoop(
            encoded: encoded,
            numEncoderFrames: numEncoderFrames,
            currentToken: &currentToken,
            currentH: &currentH,
            currentC: &currentC,
            shouldProfile: shouldProfile
        )

        self.lastToken = currentToken
        self.hState = currentH
        self.cState = currentC

        if let callback = partialCallback, let tokenizer = tokenizer, !accumulatedTokenIds.isEmpty {
            callback(tokenizer.decode(ids: accumulatedTokenIds, skipSpecialTokens: true))
        }

        processedChunks += 1
        if shouldProfile {
            componentProfile.chunks += 1
            componentProfile.totalChunkTime += profileNow() - chunkStartedAt
        }
    }

    internal func processSplitEncoderMelChunk(_ melChunk: NemotronCacheAwareMelChunk) async throws {
        let activeEncoder: MLModel
        if melChunk.isFirstChunk {
            guard let encoderInit else {
                throw ASRError.notInitialized
            }
            activeEncoder = encoderInit
        } else {
            guard let encoderStep else {
                throw ASRError.notInitialized
            }
            activeEncoder = encoderStep
        }

        guard let decoder = decoder,
            let cacheTime = cacheTime,
            let cacheLen = cacheLen,
            let cacheChannel = cacheChannel,
            var currentH = hState,
            var currentC = cState
        else {
            throw ASRError.notInitialized
        }
        guard joint != nil || jointDecision != nil else {
            throw ASRError.notInitialized
        }

        let shouldProfile = componentProfilingEnabled
        let chunkStartedAt = shouldProfile ? profileNow() : 0
        var currentToken = lastToken

        let melLen = try MLMultiArray(shape: [1], dataType: .int32)
        melLen[0] = NSNumber(value: melChunk.validMelFrames)

        var encoderFeatures: [String: MLFeatureValue] = [
            "processed_signal": MLFeatureValue(multiArray: melChunk.processedSignal),
            "processed_signal_length": MLFeatureValue(multiArray: melLen),
            "cache_last_time": MLFeatureValue(multiArray: cacheTime),
            "cache_last_channel_len": MLFeatureValue(multiArray: cacheLen),
            "cache_last_channel": MLFeatureValue(multiArray: cacheChannel),
        ]
        if config.runtimePrompt {
            encoderFeatures["prompt_vector"] = MLFeatureValue(multiArray: try createPromptVector())
        }

        let encoderInput = try MLDictionaryFeatureProvider(dictionary: encoderFeatures)
        let encoderOutput = try await activeEncoder.prediction(from: encoderInput)

        if let newCacheChannel = encoderOutput.featureValue(for: "cache_last_channel_next")?.multiArrayValue {
            self.cacheChannel = newCacheChannel
        }
        if let newCacheTime = encoderOutput.featureValue(for: "cache_last_time_next")?.multiArrayValue {
            self.cacheTime = newCacheTime
        }
        if let newCacheLen = encoderOutput.featureValue(for: "cache_last_channel_next_len")?.multiArrayValue {
            self.cacheLen = newCacheLen
        }

        guard let encoded = encoderOutput.featureValue(for: "encoded")?.multiArrayValue else {
            throw ASRError.processingFailed("Encoder failed to produce encoded output")
        }

        let numEncoderFrames = validEncoderFrameCount(from: encoderOutput, encoded: encoded)
        try await runGreedyRnntDecodeLoop(
            encoded: encoded,
            numEncoderFrames: numEncoderFrames,
            currentToken: &currentToken,
            currentH: &currentH,
            currentC: &currentC,
            shouldProfile: shouldProfile
        )

        self.lastToken = currentToken
        self.hState = currentH
        self.cState = currentC

        if let callback = partialCallback, let tokenizer = tokenizer, !accumulatedTokenIds.isEmpty {
            callback(tokenizer.decode(ids: accumulatedTokenIds, skipSpecialTokens: true))
        }

        processedChunks += 1
        if shouldProfile {
            componentProfile.chunks += 1
            componentProfile.totalChunkTime += profileNow() - chunkStartedAt
        }
    }

    internal func validEncoderFrameCount(from encoderOutput: MLFeatureProvider, encoded: MLMultiArray) -> Int {
        var numEncoderFrames = encoded.shape[2].intValue
        if let encodedLen = encoderOutput.featureValue(for: "encoded_len")?.multiArrayValue {
            numEncoderFrames = min(numEncoderFrames, encodedLen[0].intValue)
        }
        return numEncoderFrames
    }

    /// Greedy RNNT decode over `numEncoderFrames` encoder steps.
    ///
    /// When `tokenSink` is nil (streaming/default), emitted token ids are appended to the
    /// shared `accumulatedTokenIds`, preserving existing behavior. When `tokenSink` is set
    /// (offline sliding-window path), emitted tokens are reported as `(tokenId, frameIndex)`
    /// and NOT appended to `accumulatedTokenIds`, so the caller can merge windows itself.
    internal func runGreedyRnntDecodeLoop(
        encoded: MLMultiArray,
        numEncoderFrames: Int,
        currentToken: inout Int32,
        currentH: inout MLMultiArray,
        currentC: inout MLMultiArray,
        shouldProfile: Bool,
        tokenSink: ((Int, Int) -> Void)? = nil
    ) async throws {
        guard let decoder, joint != nil || jointDecision != nil else {
            throw ASRError.notInitialized
        }

        let encStep = try MLMultiArray(
            shape: [1, NSNumber(value: encoded.shape[1].intValue), 1],
            dataType: .float32
        )
        let tokenInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let tokenLen = try MLMultiArray(shape: [1], dataType: .int32)
        tokenLen[0] = 1
        let decoderInput = MutableDecoderInput(
            token: tokenInput,
            tokenLength: tokenLen,
            hIn: currentH,
            cIn: currentC,
            tokenName: "targets",
            tokenLengthName: "target_length"
        )
        let decoderPlaceholder = try MLMultiArray(
            shape: [1, NSNumber(value: config.decoderHidden), 1],
            dataType: .float32
        )
        let jointInput = MutableJointInput(
            encoderStep: encStep,
            decoderStep: decoderPlaceholder,
            encoderName: "encoded",
            decoderName: "decoder"
        )

        let decodeStartedAt = shouldProfile ? profileNow() : 0
        let decodeInterval = beginProfileInterval("Nemotron.DecodeLoop")
        for t in 0..<numEncoderFrames {
            let encoderStepCopyStartedAt = shouldProfile ? profileNow() : 0
            let encoderStepCopyInterval = beginProfileInterval("Nemotron.EncoderStepCopy")
            try copyEncoderStep(from: encoded, timeIndex: t, into: encStep)
            if shouldProfile {
                componentProfile.encoderStepCopyTime += profileNow() - encoderStepCopyStartedAt
            }
            endProfileInterval("Nemotron.EncoderStepCopy", encoderStepCopyInterval)

            for _ in 0..<10 {
                tokenInput[0] = NSNumber(value: currentToken)
                decoderInput.hIn = currentH
                decoderInput.cIn = currentC

                let decoderStartedAt = shouldProfile ? profileNow() : 0
                let decoderInterval = beginProfileInterval("Nemotron.Decoder")
                let decoderOutput = try await decoder.prediction(from: decoderInput)
                if shouldProfile {
                    componentProfile.decoderTime += profileNow() - decoderStartedAt
                }
                endProfileInterval("Nemotron.Decoder", decoderInterval)
                guard let decoderOut = decoderOutput.featureValue(for: "decoder")?.multiArrayValue,
                    let hOut = decoderOutput.featureValue(for: "h_out")?.multiArrayValue,
                    let cOut = decoderOutput.featureValue(for: "c_out")?.multiArrayValue
                else {
                    throw ASRError.processingFailed("Decoder failed")
                }

                jointInput.decoderStep = decoderOut

                let predToken: Int
                if let jointDecision {
                    let jointStartedAt = shouldProfile ? profileNow() : 0
                    let jointDecisionInterval = beginProfileInterval("Nemotron.JointDecision")
                    let decisionOutput = try await jointDecision.prediction(from: jointInput)
                    if shouldProfile {
                        componentProfile.jointDecisionTime += profileNow() - jointStartedAt
                    }
                    endProfileInterval("Nemotron.JointDecision", jointDecisionInterval)
                    guard let tokenId = decisionOutput.featureValue(for: "token_id")?.multiArrayValue else {
                        throw ASRError.processingFailed("Joint decision failed")
                    }
                    predToken = tokenId[0].intValue
                } else {
                    guard let joint else {
                        throw ASRError.processingFailed("Missing joint model")
                    }
                    let jointStartedAt = shouldProfile ? profileNow() : 0
                    let jointInterval = beginProfileInterval("Nemotron.Joint")
                    let jointOutput = try await joint.prediction(from: jointInput)
                    if shouldProfile {
                        componentProfile.jointTime += profileNow() - jointStartedAt
                    }
                    endProfileInterval("Nemotron.Joint", jointInterval)
                    guard let logits = jointOutput.featureValue(for: "logits")?.multiArrayValue else {
                        throw ASRError.processingFailed("Joint failed")
                    }
                    predToken = findMaxIndex(logits)
                }
                if shouldProfile {
                    componentProfile.decodeSteps += 1
                }

                if predToken == config.blankIdx {
                    break
                }

                if let tokenSink {
                    tokenSink(predToken, t)
                } else {
                    accumulatedTokenIds.append(predToken)
                }
                currentToken = Int32(predToken)
                currentH = hOut
                currentC = cOut
            }
        }
        if shouldProfile {
            componentProfile.decodeLoopTime += profileNow() - decodeStartedAt
        }
        endProfileInterval("Nemotron.DecodeLoop", decodeInterval)
    }

    // MARK: - Tensor Utilities

    internal func createAudioArray(_ samples: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: samples.count)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: samples.count)
        ptr.update(from: samples, count: samples.count)
        return array
    }

    internal func createPaddedAudioArray(_ samples: [Float], count: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: count)], dataType: .float32)
        array.reset(to: 0)
        let copyCount = min(samples.count, count)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        ptr.update(from: samples, count: copyCount)
        return array
    }

    internal func createPromptVector() throws -> MLMultiArray {
        if let promptVectorCache {
            return promptVectorCache
        }

        let language = activeTargetLanguage ?? config.targetLang ?? "auto"
        guard let promptIndex = config.promptDictionary[language] else {
            let available = config.promptDictionary.keys.sorted().prefix(12).joined(separator: ", ")
            throw ASRError.processingFailed("Unknown Nemotron prompt language '\(language)'. Available: \(available)")
        }
        guard promptIndex >= 0 && promptIndex < config.numPrompts else {
            throw ASRError.processingFailed(
                "Prompt index \(promptIndex) for '\(language)' is outside num_prompts=\(config.numPrompts)")
        }

        let array = try MLMultiArray(shape: [1, NSNumber(value: config.numPrompts)], dataType: .float32)
        array.reset(to: 0)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        ptr[promptIndex] = 1.0
        promptVectorCache = array
        return array
    }

    internal func buildSingleEncoderMelInput(
        from processedSignal: MLMultiArray,
        totalMelFrames: Int? = nil,
        preEncodeCache: Int? = nil
    ) throws -> MLMultiArray {
        let totalFrames = totalMelFrames ?? config.totalMelFrames
        let cacheFramesBudget = preEncodeCache ?? config.preEncodeCache
        let result = try MLMultiArray(
            shape: [1, NSNumber(value: config.melFeatures), NSNumber(value: totalFrames)],
            dataType: .float32
        )
        result.reset(to: 0)

        let resultPtr = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)
        let resultStride0 = result.strides[0].intValue
        let resultStride1 = result.strides[1].intValue
        let resultStride2 = result.strides[2].intValue

        if let melCache = melCache {
            let cacheFrames = min(melCache.shape[2].intValue, cacheFramesBudget)
            let cachePtr = melCache.dataPointer.bindMemory(to: Float.self, capacity: melCache.count)
            let cacheStride0 = melCache.strides[0].intValue
            let cacheStride1 = melCache.strides[1].intValue
            let cacheStride2 = melCache.strides[2].intValue

            for mel in 0..<config.melFeatures {
                for t in 0..<cacheFrames {
                    let srcIdx = 0 * cacheStride0 + mel * cacheStride1 + t * cacheStride2
                    let dstIdx = 0 * resultStride0 + mel * resultStride1 + t * resultStride2
                    resultPtr[dstIdx] = cachePtr[srcIdx]
                }
            }
        }

        let signalFrames = processedSignal.shape[2].intValue
        let copyFrames = min(config.chunkMelFrames, signalFrames, totalFrames - cacheFramesBudget)
        let signalPtr = processedSignal.dataPointer.bindMemory(to: Float.self, capacity: processedSignal.count)
        let signalStride0 = processedSignal.strides[0].intValue
        let signalStride1 = processedSignal.strides[1].intValue
        let signalStride2 = processedSignal.strides[2].intValue

        for mel in 0..<config.melFeatures {
            for t in 0..<copyFrames {
                let srcIdx = 0 * signalStride0 + mel * signalStride1 + t * signalStride2
                let dstIdx = 0 * resultStride0 + mel * resultStride1 + (cacheFramesBudget + t) * resultStride2
                resultPtr[dstIdx] = signalPtr[srcIdx]
            }
        }

        return result
    }

    internal func prependMelCache(to chunkMel: MLMultiArray) throws -> MLMultiArray {
        // Prepend cached mel frames (9) to current chunk mel (112) → [1, 128, 121]
        // Input: chunkMel [1, 128, ~112]
        // Output: [1, 128, 121] = 9 cache + 112 chunk (or padded)

        let chunkFrames = chunkMel.shape[2].intValue
        let totalFrames = config.totalMelFrames

        let result = try MLMultiArray(
            shape: [1, NSNumber(value: config.melFeatures), NSNumber(value: totalFrames)],
            dataType: .float32
        )
        result.reset(to: 0)

        let resultPtr = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)
        let chunkPtr = chunkMel.dataPointer.bindMemory(to: Float.self, capacity: chunkMel.count)

        let resultStride0 = result.strides[0].intValue
        let resultStride1 = result.strides[1].intValue
        let resultStride2 = result.strides[2].intValue
        let chunkStride0 = chunkMel.strides[0].intValue
        let chunkStride1 = chunkMel.strides[1].intValue
        let chunkStride2 = chunkMel.strides[2].intValue

        // Copy mel cache (or zeros if first chunk)
        if let melCache = melCache {
            let cachePtr = melCache.dataPointer.bindMemory(to: Float.self, capacity: melCache.count)
            let cacheFrames = melCache.shape[2].intValue
            let cacheStride0 = melCache.strides[0].intValue
            let cacheStride1 = melCache.strides[1].intValue
            let cacheStride2 = melCache.strides[2].intValue

            for mel in 0..<config.melFeatures {
                for t in 0..<cacheFrames {
                    let srcIdx = 0 * cacheStride0 + mel * cacheStride1 + t * cacheStride2
                    let dstIdx = 0 * resultStride0 + mel * resultStride1 + t * resultStride2
                    resultPtr[dstIdx] = cachePtr[srcIdx]
                }
            }
        }

        // Copy chunk mel (after cache position)
        let copyFrames = min(chunkFrames, totalFrames - config.preEncodeCache)
        for mel in 0..<config.melFeatures {
            for t in 0..<copyFrames {
                let srcIdx = 0 * chunkStride0 + mel * chunkStride1 + t * chunkStride2
                let dstIdx = 0 * resultStride0 + mel * resultStride1 + (config.preEncodeCache + t) * resultStride2
                resultPtr[dstIdx] = chunkPtr[srcIdx]
            }
        }

        return result
    }

    internal func extractMelCache(from chunkMel: MLMultiArray) throws -> MLMultiArray {
        // Extract last preEncodeCache (9) frames from chunk mel
        let chunkFrames = chunkMel.shape[2].intValue
        let cacheFrames = min(config.preEncodeCache, chunkFrames)

        let cache = try MLMultiArray(
            shape: [1, NSNumber(value: config.melFeatures), NSNumber(value: cacheFrames)],
            dataType: .float32
        )

        let srcPtr = chunkMel.dataPointer.bindMemory(to: Float.self, capacity: chunkMel.count)
        let dstPtr = cache.dataPointer.bindMemory(to: Float.self, capacity: cache.count)

        let srcStride0 = chunkMel.strides[0].intValue
        let srcStride1 = chunkMel.strides[1].intValue
        let srcStride2 = chunkMel.strides[2].intValue
        let dstStride0 = cache.strides[0].intValue
        let dstStride1 = cache.strides[1].intValue
        let dstStride2 = cache.strides[2].intValue

        let startT = chunkFrames - cacheFrames

        for mel in 0..<config.melFeatures {
            for t in 0..<cacheFrames {
                let srcIdx = 0 * srcStride0 + mel * srcStride1 + (startT + t) * srcStride2
                let dstIdx = 0 * dstStride0 + mel * dstStride1 + t * dstStride2
                dstPtr[dstIdx] = srcPtr[srcIdx]
            }
        }

        return cache
    }

    internal func extractEncoderStep(from encoded: MLMultiArray, timeIndex: Int) throws -> MLMultiArray {
        // encoded: [1, 1024, T] -> step: [1, 1024, 1]
        let dim = encoded.shape[1].intValue
        let step = try MLMultiArray(shape: [1, NSNumber(value: dim), 1], dataType: .float32)
        try copyEncoderStep(from: encoded, timeIndex: timeIndex, into: step)
        return step
    }

    internal func copyEncoderStep(from encoded: MLMultiArray, timeIndex: Int, into step: MLMultiArray) throws {
        // encoded: [1, C, T] -> step: [1, C, 1]
        let dim = encoded.shape[1].intValue
        guard step.count >= dim else {
            throw ASRError.processingFailed("Encoder step backing is too small")
        }
        let srcPtr = encoded.dataPointer.bindMemory(to: Float.self, capacity: encoded.count)
        let dstPtr = step.dataPointer.bindMemory(to: Float.self, capacity: step.count)

        let stride0 = encoded.strides[0].intValue
        let stride1 = encoded.strides[1].intValue
        let stride2 = encoded.strides[2].intValue

        for c in 0..<dim {
            let srcIdx = 0 * stride0 + c * stride1 + timeIndex * stride2
            dstPtr[c] = srcPtr[srcIdx]
        }
    }

    internal func sliceDecoderOutput(_ decoderOut: MLMultiArray) throws -> MLMultiArray {
        // decoder_out: [1, hidden, T] -> [1, hidden, 1] (first frame, index 0)
        // Python uses decoder_out[:, :, :1] which is the FIRST frame
        let hidden = decoderOut.shape[1].intValue

        let result = try MLMultiArray(shape: [1, NSNumber(value: hidden), 1], dataType: .float32)

        let srcPtr = decoderOut.dataPointer.bindMemory(to: Float.self, capacity: decoderOut.count)
        let dstPtr = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)

        let stride0 = decoderOut.strides[0].intValue
        let stride1 = decoderOut.strides[1].intValue
        let stride2 = decoderOut.strides[2].intValue

        // Use FIRST frame (index 0), not last frame
        let firstT = 0
        for c in 0..<hidden {
            let srcIdx = 0 * stride0 + c * stride1 + firstT * stride2
            dstPtr[c] = srcPtr[srcIdx]
        }

        return result
    }

    internal func findMaxIndex(_ logits: MLMultiArray) -> Int {
        // logits: [1, 1, 1, vocab_size+1]
        // Use actual logits count to prevent out-of-bounds when config is incorrect
        let count = logits.count

        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: count)

        // Use Accelerate framework for vectorized maximum index search
        var maxVal: Float = -Float.infinity
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(ptr, 1, &maxVal, &maxIdx, vDSP_Length(count))

        return Int(maxIdx)
    }
}
