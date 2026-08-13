@preconcurrency import CoreML
import Foundation
import OSLog

private let granitePlusLogger = Logger(subsystem: "FluidAudio", category: "GranitePlusAsrModels")

public enum GranitePlusTask: String, Sendable {
    case asr
    case timestamp
}

public struct GranitePlusAudioWindowMeta: Codable, Sendable {
    public let seconds: Int
    public let samples: Int
    public let frames: Int
    public let audioTokens: Int
    public let package: String
    public let inputs: [String]
    public let outputs: [String]

    private enum CodingKeys: String, CodingKey {
        case seconds
        case samples
        case frames
        case audioTokens = "audio_tokens"
        case package
        case inputs
        case outputs
    }
}

public struct GranitePlusManifest: Codable, Sendable {
    public let modelID: String
    public let sampleRate: Int
    public let nFFT: Int
    public let winLength: Int
    public let hopLength: Int
    public let nMels: Int
    public let featureFramesPerSecond: Int
    public let melFilters: String
    public let tokenizer: String
    public let tokenEmbeddingPackage: String
    public let languageModelPackage: String
    public let maxSequenceLength: Int
    public let maxQueryLength: Int
    public let audioTokenID: Int
    public let eosTokenID: Int
    public let defaultWindowSeconds: Int
    public let windows: [String: GranitePlusAudioWindowMeta]

    private enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case sampleRate = "sample_rate"
        case nFFT = "n_fft"
        case winLength = "win_length"
        case hopLength = "hop_length"
        case nMels = "n_mels"
        case featureFramesPerSecond = "feature_frames_per_second"
        case melFilters = "mel_filters"
        case tokenizer
        case tokenEmbeddingPackage = "token_embedding_package"
        case languageModelPackage = "language_model_package"
        case maxSequenceLength = "max_sequence_length"
        case maxQueryLength = "max_query_length"
        case audioTokenID = "audio_token_id"
        case eosTokenID = "eos_token_id"
        case defaultWindowSeconds = "default_window_seconds"
        case windows
    }
}

@available(macOS 15, iOS 18, *)
public struct GranitePlusAsrModels: @unchecked Sendable {
    public static let manifestFile = "granite_plus_manifest.json"

    public let audioModel: MLModel
    public let tokenEmbeddingModel: MLModel
    public let languageModel: MLModel
    public let manifest: GranitePlusManifest
    public let tokenizer: GraniteTokenizer
    public let modelDirectory: URL
    public let computeUnits: MLComputeUnits

    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> GranitePlusAsrModels {
        let manifest = try loadManifest(from: directory)
        let tokenizer = try await GraniteTokenizer(modelDirectory: directory)
        let windowKey = "\(manifest.defaultWindowSeconds)s"
        guard let audioWindow = manifest.windows[windowKey] else {
            throw GraniteAsrError.modelNotFound("Granite Plus window \(windowKey)")
        }

        let audioModel = try await loadPackage(named: audioWindow.package, from: directory, computeUnits: computeUnits)
        let tokenEmbeddingModel = try await loadPackage(
            named: manifest.tokenEmbeddingPackage,
            from: directory,
            computeUnits: computeUnits
        )
        let languageModel = try await loadPackage(
            named: manifest.languageModelPackage,
            from: directory,
            computeUnits: computeUnits
        )

        granitePlusLogger.info("Loaded Granite Plus CoreML models from \(directory.path, privacy: .public)")
        return GranitePlusAsrModels(
            audioModel: audioModel,
            tokenEmbeddingModel: tokenEmbeddingModel,
            languageModel: languageModel,
            manifest: manifest,
            tokenizer: tokenizer,
            modelDirectory: directory,
            computeUnits: computeUnits
        )
    }

    public static func modelsExist(at directory: URL) -> Bool {
        guard let manifest = try? loadManifest(from: directory) else {
            return false
        }
        let fileManager = FileManager.default
        let requiredFiles = [
            manifest.melFilters,
            manifest.tokenizer,
            manifest.tokenEmbeddingPackage,
            manifest.languageModelPackage
        ]
        let hasRequiredFiles = requiredFiles.allSatisfy {
            fileManager.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        guard hasRequiredFiles else {
            return false
        }
        return manifest.windows.values.allSatisfy { meta in
            fileManager.fileExists(atPath: directory.appendingPathComponent(meta.package).path)
        }
    }

    func featureManifest() -> GraniteAsrManifest {
        let windows = manifest.windows.mapValues { meta in
            GraniteWindowMeta(
                seconds: meta.seconds,
                samples: meta.samples,
                frames: meta.frames,
                bpeFrames: meta.audioTokens,
                package: meta.package,
                inputs: meta.inputs,
                outputs: meta.outputs
            )
        }
        return GraniteAsrManifest(
            modelID: manifest.modelID,
            sampleRate: manifest.sampleRate,
            nFFT: manifest.nFFT,
            winLength: manifest.winLength,
            hopLength: manifest.hopLength,
            nMels: manifest.nMels,
            featureFramesPerSecond: manifest.featureFramesPerSecond,
            bpePoolingWindow: 4,
            defaultWindowSeconds: manifest.defaultWindowSeconds,
            defaultOverlapSeconds: 0,
            speedWindowSeconds: manifest.defaultWindowSeconds,
            speedOverlapSeconds: 0,
            melFilters: manifest.melFilters,
            tokenizer: manifest.tokenizer,
            windows: windows
        )
    }

    private static func loadManifest(from directory: URL) throws -> GranitePlusManifest {
        let url = directory.appendingPathComponent(Self.manifestFile)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GraniteAsrError.manifestMissing(url)
        }

        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(GranitePlusManifest.self, from: data)
        } catch {
            throw GraniteAsrError.invalidManifest(url, error)
        }
    }

    private static func loadPackage(
        named packageName: String,
        from directory: URL,
        computeUnits: MLComputeUnits
    ) async throws -> MLModel {
        try await GraniteAsrModels.loadWindowModel(
            GraniteWindowMeta(
                seconds: 0,
                samples: 0,
                frames: 0,
                bpeFrames: 0,
                package: packageName,
                inputs: [],
                outputs: []
            ),
            from: directory,
            computeUnits: computeUnits
        )
    }
}

enum GranitePlusPromptBuilder {
    private static let systemContent = """
    Knowledge Cutoff Date: April 2024.
    Today's Date: December 19, 2024.
    You are Granite, developed by IBM. You are a helpful AI assistant
    """

    static func prompt(task: GranitePlusTask, audioTokenCount: Int) -> String {
        let audioTokens = String(repeating: "<|audio|>", count: audioTokenCount)
        let userPrompt: String
        switch task {
        case .asr:
            userPrompt = "\(audioTokens) can you transcribe the speech into a written format?"
        case .timestamp:
            userPrompt = "\(audioTokens) Timestamps: Transcribe the speech. After each word, "
                + "add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
        }
        return "<|start_of_role|>system<|end_of_role|>\(systemContent)<|end_of_text|>\n"
            + "<|start_of_role|>user<|end_of_role|>\(userPrompt)<|end_of_text|>\n"
            + "<|start_of_role|>assistant<|end_of_role|>"
    }
}
