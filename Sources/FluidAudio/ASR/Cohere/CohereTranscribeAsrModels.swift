@preconcurrency import CoreML
import Foundation
import OSLog

private let cohereLogger = Logger(subsystem: "FluidAudio", category: "CohereTranscribeAsrModels")

public enum CohereTranscribeAsrError: LocalizedError {
    case manifestMissing(URL)
    case invalidManifest(URL, Error)
    case modelNotFound(String)
    case invalidOutput(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .manifestMissing(let url):
            return "Cohere manifest not found at \(url.path)"
        case .invalidManifest(let url, let error):
            return "Failed to decode Cohere manifest at \(url.path): \(error.localizedDescription)"
        case .modelNotFound(let name):
            return "Required Cohere CoreML model not found: \(name)"
        case .invalidOutput(let message):
            return "Invalid Cohere model output: \(message)"
        case .generationFailed(let message):
            return "Cohere generation failed: \(message)"
        }
    }
}

public struct CohereTranscribeAsrManifest: Codable, Sendable {
    public let modelID: String
    public let sampleRate: Int
    public let maxAudioSamples: Int
    public let maxAudioSeconds: Double
    public let overlapSeconds: Double
    public let overlapSamples: Int
    public let maxFeatureFrames: Int
    public let maxEncoderFrames: Int
    public let encoderHiddenSize: Int
    public let decoderMaxLen: Int
    public let defaultMaxNewTokens: Int
    public let promptIDs: [Int]
    public let eosTokenID: Int
    public let padTokenID: Int
    public let idToToken: [String]

    private enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case sampleRate = "sample_rate"
        case maxAudioSamples = "max_audio_samples"
        case maxAudioSeconds = "max_audio_seconds"
        case overlapSeconds = "overlap_seconds"
        case overlapSamples = "overlap_samples"
        case maxFeatureFrames = "max_feature_frames"
        case maxEncoderFrames = "max_encoder_frames"
        case encoderHiddenSize = "encoder_hidden_size"
        case decoderMaxLen = "decoder_max_len"
        case defaultMaxNewTokens = "default_max_new_tokens"
        case promptIDs = "prompt_ids"
        case eosTokenID = "eos_token_id"
        case padTokenID = "pad_token_id"
        case idToToken = "id_to_token"
    }
}

@available(macOS 15, iOS 18, *)
public struct CohereTranscribeAsrModels: Sendable {
    public static let manifestFile = "coreml_manifest.json"
    public static let frontendFile = "cohere_frontend.mlpackage"
    public static let encoderFile = "cohere_encoder.mlpackage"
    public static let decoderFile = "cohere_decoder_fullseq_masked.mlpackage"
    public static let cachedDecoderFile = "cohere_decoder_cached.mlpackage"

    public let frontend: MLModel
    public let encoder: MLModel
    public let decoder: MLModel
    public let cachedDecoder: MLModel
    public let manifest: CohereTranscribeAsrManifest

    public static func modelsExist(at directory: URL) -> Bool {
        let required = [
            Self.manifestFile,
            Self.frontendFile,
            Self.encoderFile,
            Self.decoderFile,
            Self.cachedDecoderFile,
        ]
        return required.allSatisfy { name in
            FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path)
        }
    }

    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> CohereTranscribeAsrModels {
        let manifest = try self.loadManifest(from: directory)

        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        configuration.allowLowPrecisionAccumulationOnGPU = true

        async let frontend = self.loadPackage(Self.frontendFile, from: directory, configuration: configuration)
        async let encoder = self.loadPackage(Self.encoderFile, from: directory, configuration: configuration)
        async let decoder = self.loadPackage(Self.decoderFile, from: directory, configuration: configuration)
        async let cachedDecoder = self.loadPackage(Self.cachedDecoderFile, from: directory, configuration: configuration)

        return try await CohereTranscribeAsrModels(
            frontend: frontend,
            encoder: encoder,
            decoder: decoder,
            cachedDecoder: cachedDecoder,
            manifest: manifest
        )
    }

    public static func compiledArtifactsDirectory(for sourceDirectory: URL) -> URL {
        let cachesDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? sourceDirectory.deletingLastPathComponent()
        let stableName = self.stableCompiledDirectoryName(for: sourceDirectory)
        return cachesDirectory
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("CompiledCohereModels", isDirectory: true)
            .appendingPathComponent(stableName, isDirectory: true)
    }

    private static func loadManifest(from directory: URL) throws -> CohereTranscribeAsrManifest {
        let url = directory.appendingPathComponent(Self.manifestFile)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw CohereTranscribeAsrError.manifestMissing(url)
        }

        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(CohereTranscribeAsrManifest.self, from: data)
        } catch {
            throw CohereTranscribeAsrError.invalidManifest(url, error)
        }
    }

    private static func loadPackage(
        _ packageName: String,
        from directory: URL,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        let packageURL = directory.appendingPathComponent(packageName)
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw CohereTranscribeAsrError.modelNotFound(packageName)
        }

        let compiledName = packageName.replacingOccurrences(of: ".mlpackage", with: ".mlmodelc")
        let compiledRootDirectory = self.compiledArtifactsDirectory(for: directory)
        try FileManager.default.createDirectory(at: compiledRootDirectory, withIntermediateDirectories: true)
        let compiledURL = compiledRootDirectory.appendingPathComponent(compiledName, isDirectory: true)
        let modelURL: URL

        if FileManager.default.fileExists(atPath: compiledURL.path) {
            modelURL = compiledURL
        } else {
            cohereLogger.info("Compiling \(packageName, privacy: .public)")
            let tempCompiledURL = try await MLModel.compileModel(at: packageURL)
            try? FileManager.default.removeItem(at: compiledURL)
            try FileManager.default.copyItem(at: tempCompiledURL, to: compiledURL)
            try? FileManager.default.removeItem(at: tempCompiledURL)
            modelURL = compiledURL
        }

        return try await MLModel.load(contentsOf: modelURL, configuration: configuration)
    }

    private static func stableCompiledDirectoryName(for sourceDirectory: URL) -> String {
        let path = sourceDirectory.standardizedFileURL.path
        let folderName = sourceDirectory.lastPathComponent.replacingOccurrences(of: " ", with: "_")
        return "\(folderName)-\(self.fnv1a64(path))"
    }

    private static func fnv1a64(_ string: String) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        let prime: UInt64 = 0x100000001b3

        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash &*= prime
        }

        return String(hash, radix: 16, uppercase: false)
    }
}
