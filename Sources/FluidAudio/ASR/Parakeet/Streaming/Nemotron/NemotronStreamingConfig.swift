import Foundation

/// Configuration for Nemotron Speech Streaming 0.6B
/// Loaded from metadata.json for each chunk size variant
public struct NemotronStreamingConfig: Sendable {
    public enum ModelLayout: Sendable {
        case legacy
        case singleEncoder
    }

    /// Sample rate in Hz
    public let sampleRate: Int
    /// Number of mel spectrogram features
    public let melFeatures: Int
    /// Mel frames per chunk
    public let chunkMelFrames: Int
    /// Chunk duration in milliseconds
    public let chunkMs: Int
    /// Mel frames to advance after each processed chunk
    public let shiftMelFrames: Int
    /// Pre-encode cache size in mel frames (for encoder context)
    public let preEncodeCache: Int
    /// Total mel frames for encoder input (cache + chunk)
    public let totalMelFrames: Int
    /// Vocabulary size
    public let vocabSize: Int
    /// Blank token index (== vocab_size)
    public let blankIdx: Int
    /// Encoder output dimension
    public let encoderDim: Int
    /// Decoder hidden size
    public let decoderHidden: Int
    /// Number of decoder LSTM layers
    public let decoderLayers: Int
    /// Encoder cache shapes
    public let cacheChannelShape: [Int]
    public let cacheTimeShape: [Int]
    public let modelLayout: ModelLayout
    public let maxAudioSamples: Int?
    public let runtimePrompt: Bool
    public let promptDictionary: [String: Int]
    public let numPrompts: Int
    public let targetLang: String?
    public let encoderCacheFloat16: Bool

    /// Audio samples per chunk
    public var chunkSamples: Int { chunkMelFrames * 160 }
    /// Audio samples to advance after each processed chunk
    public var shiftSamples: Int { shiftMelFrames * 160 }

    /// Default config for 1120ms chunks (backward compatibility)
    public init() {
        self.sampleRate = 16000
        self.melFeatures = 128
        self.chunkMelFrames = 112
        self.chunkMs = 1120
        self.shiftMelFrames = 112
        self.preEncodeCache = 9
        self.totalMelFrames = 121
        self.vocabSize = 1024
        self.blankIdx = 1024
        self.encoderDim = 1024
        self.decoderHidden = 640
        self.decoderLayers = 2
        self.cacheChannelShape = [1, 24, 70, 1024]
        self.cacheTimeShape = [1, 24, 1024, 8]
        self.modelLayout = .legacy
        self.maxAudioSamples = nil
        self.runtimePrompt = false
        self.promptDictionary = [:]
        self.numPrompts = 128
        self.targetLang = nil
        self.encoderCacheFloat16 = false
    }

    /// Load config from metadata.json
    public init(from metadataURL: URL) throws {
        let data = try Data(contentsOf: metadataURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ASRError.processingFailed("Invalid metadata.json format")
        }

        let shapes = json["shapes"] as? [String: [Int]] ?? [:]
        let components = (json["coreml"] as? [String: Any])?["components"] as? [String: String] ?? [:]
        let processedStepShape = shapes["processed_signal_step"]
        let singleEncoder = components["encoder"] != nil && processedStepShape != nil

        let preEncodeCache = json["pre_encode_cache"] as? Int ?? 9
        let totalMelFrames = json["total_mel_frames"] as? Int ?? processedStepShape?[2] ?? 121
        let chunkMelFrames = json["chunk_mel_frames"] as? Int
            ?? (singleEncoder ? max(totalMelFrames - preEncodeCache, 1) : 112)
        let streaming = json["streaming"] as? [String: Any]
        let shiftSize = streaming?["shift_size"] as? [Int]
        let shiftMelFrames = json["shift_mel_frames"] as? Int ?? shiftSize?.last ?? chunkMelFrames
        self.sampleRate = json["sample_rate"] as? Int ?? 16000
        self.melFeatures = json["mel_features"] as? Int ?? processedStepShape?[1] ?? 128
        self.chunkMelFrames = chunkMelFrames
        self.chunkMs = json["chunk_ms"] as? Int ?? (shiftMelFrames * 10)
        self.shiftMelFrames = shiftMelFrames
        self.preEncodeCache = preEncodeCache
        self.totalMelFrames = totalMelFrames
        self.vocabSize = json["vocab_size"] as? Int ?? 1024
        self.blankIdx = json["blank_idx"] as? Int ?? 1024
        self.encoderDim = json["encoder_dim"] as? Int ?? shapes["encoded_step"]?[1] ?? 1024
        self.decoderHidden = json["decoder_hidden"] as? Int ?? 640
        self.decoderLayers = json["decoder_layers"] as? Int ?? 2
        self.cacheChannelShape = json["cache_channel_shape"] as? [Int] ?? shapes["cache_last_channel"] ?? [1, 24, 70, 1024]
        self.cacheTimeShape = json["cache_time_shape"] as? [Int] ?? shapes["cache_last_time"] ?? [1, 24, 1024, 8]
        self.modelLayout = singleEncoder ? .singleEncoder : .legacy
        let coreML = json["coreml"] as? [String: Any]
        let flexiblePreprocessor = coreML?["preprocessor_audio_flexible"] as? Bool ?? false
        self.maxAudioSamples = flexiblePreprocessor ? nil : (json["max_audio_samples"] as? Int ?? shapes["audio_signal"]?[1])
        self.runtimePrompt = json["runtime_prompt"] as? Bool ?? false
        self.promptDictionary = Self.parsePromptDictionary(json["prompt_dictionary"])
        self.numPrompts = json["num_prompts"] as? Int ?? 128
        self.targetLang = json["target_lang"] as? String
        let encoderCacheIO = (coreML?["encoder_cache_io"] as? String ?? json["encoder_cache_io"] as? String ?? "FLOAT32")
            .uppercased()
        self.encoderCacheFloat16 = encoderCacheIO == "FLOAT16" || encoderCacheIO == "FP16"
    }

    private static func parsePromptDictionary(_ value: Any?) -> [String: Int] {
        guard let dictionary = value as? [String: Any] else {
            return [:]
        }

        var result: [String: Int] = [:]
        result.reserveCapacity(dictionary.count)
        for (key, value) in dictionary {
            if let intValue = value as? Int {
                result[key] = intValue
            } else if let numberValue = value as? NSNumber {
                result[key] = numberValue.intValue
            }
        }
        return result
    }
}
