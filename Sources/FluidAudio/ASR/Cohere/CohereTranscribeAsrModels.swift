@preconcurrency import CoreML
import Foundation
import OSLog

private let cohereLogger = Logger(subsystem: "FluidAudio", category: "CohereTranscribeAsrModels")

public enum CohereTranscribeAsrError: LocalizedError {
    case manifestMissing(URL)
    case invalidManifest(URL, Error)
    case modelNotFound(String)
    case invalidModelContract(String)
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
        case .invalidModelContract(let message):
            return "Invalid Cohere model contract: \(message)"
        case .invalidOutput(let message):
            return "Invalid Cohere model output: \(message)"
        case .generationFailed(let message):
            return "Cohere generation failed: \(message)"
        }
    }
}

public struct CohereTranscribeStageMeta: Codable, Sendable {
    public let package: String
    public let inputs: [String]
    public let outputs: [String]
}

public struct CohereTranscribeCachedDecoderMeta: Codable, Sendable {
    public let package: String
    public let inputs: [String]
    public let outputs: [String]
    public let logitsOutput: String
    public let cacheKOutput: String?
    public let cacheVOutput: String?
    public let numLayers: Int
    public let numHeads: Int
    public let headDim: Int

    private enum CodingKeys: String, CodingKey {
        case package
        case inputs
        case outputs
        case logitsOutput = "logits_output"
        case cacheKOutput = "cache_k_output"
        case cacheVOutput = "cache_v_output"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case headDim = "head_dim"
    }
}

public struct CohereTranscribeCrossKVMeta: Codable, Sendable {
    public let package: String
    public let inputs: [String]
    public let outputs: [String]
    public let crossKOutput: String?
    public let crossVOutput: String?

    private enum CodingKeys: String, CodingKey {
        case package
        case inputs
        case outputs
        case crossKOutput = "cross_k_output"
        case crossVOutput = "cross_v_output"
    }
}

public struct CohereTranscribeAsrManifest: Codable, Sendable {
    public let modelID: String
    public let sampleRate: Int
    public let precision: String?
    public let quantize: String?
    public let preemph: Float?
    public let maxAudioSamples: Int
    public let maxAudioSeconds: Double?
    public let overlapSeconds: Double?
    public let overlapSamples: Int?
    public let maxFeatureFrames: Int
    public let maxEncoderFrames: Int
    public let decoderMaxLen: Int
    public let defaultMaxNewTokens: Int
    public let promptIDs: [Int]
    public let eosTokenID: Int?
    public let padTokenID: Int?
    public let idToToken: [String]
    public let frontend: CohereTranscribeStageMeta
    public let encoder: CohereTranscribeStageMeta
    public let decoder: CohereTranscribeStageMeta
    public let crossKVProjector: CohereTranscribeCrossKVMeta?
    public let decoderCached: CohereTranscribeCachedDecoderMeta?

    public var isFp16: Bool { self.precision == "float16" }

    private enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case sampleRate = "sample_rate"
        case precision
        case quantize
        case preemph
        case maxAudioSamples = "max_audio_samples"
        case maxAudioSeconds = "max_audio_seconds"
        case overlapSeconds = "overlap_seconds"
        case overlapSamples = "overlap_samples"
        case maxFeatureFrames = "max_feature_frames"
        case maxEncoderFrames = "max_encoder_frames"
        case decoderMaxLen = "decoder_max_len"
        case defaultMaxNewTokens = "default_max_new_tokens"
        case promptIDs = "prompt_ids"
        case eosTokenID = "eos_token_id"
        case padTokenID = "pad_token_id"
        case idToToken = "id_to_token"
        case frontend
        case encoder
        case decoder
        case crossKVProjector = "cross_kv_projector"
        case decoderCached = "decoder_cached"
    }
}

public struct CohereTranscribeComputeConfiguration: Sendable {
    public let frontend: MLComputeUnits
    public let encoder: MLComputeUnits
    public let crossKV: MLComputeUnits
    public let decoder: MLComputeUnits

    public init(
        frontend: MLComputeUnits,
        encoder: MLComputeUnits,
        crossKV: MLComputeUnits,
        decoder: MLComputeUnits
    ) {
        self.frontend = frontend
        self.encoder = encoder
        self.crossKV = crossKV
        self.decoder = decoder
    }

    public init(uniform computeUnits: MLComputeUnits) {
        self.init(
            frontend: computeUnits,
            encoder: computeUnits,
            crossKV: computeUnits,
            decoder: computeUnits
        )
    }

    public static let sequentialGPU = Self(uniform: .cpuAndGPU)
    public static let aneSmall = Self(
        frontend: .cpuAndNeuralEngine,
        encoder: .cpuAndGPU,
        crossKV: .cpuAndNeuralEngine,
        decoder: .cpuAndNeuralEngine
    )

    public var usesSplitCompute: Bool {
        self.frontend != self.encoder
            || self.frontend != self.crossKV
            || self.frontend != self.decoder
    }
}

@available(macOS 15, iOS 18, *)
public struct CohereTranscribeAsrModels: Sendable {
    public static let manifestFile = "coreml_manifest.json"

    public let frontend: MLModel
    public let encoder: MLModel
    public let decoder: MLModel
    public let crossKVProjector: MLModel?
    public let cachedDecoder: MLModel?
    public let manifest: CohereTranscribeAsrManifest
    public let computeConfiguration: CohereTranscribeComputeConfiguration

    public static func modelsExist(at directory: URL) -> Bool {
        guard let manifest = try? self.loadManifest(from: directory) else {
            return self.legacyRequiredFiles.allSatisfy { name in
                FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path)
            }
        }

        let required = [
            manifest.frontend.package,
            manifest.encoder.package,
            manifest.decoder.package,
            manifest.crossKVProjector?.package,
            manifest.decoderCached?.package,
        ]
            .compactMap { $0 }

        return required.allSatisfy { name in
            FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path)
        }
    }

    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> CohereTranscribeAsrModels {
        try await self.load(
            from: directory,
            computeConfiguration: .init(uniform: computeUnits)
        )
    }

    public static func load(
        from directory: URL,
        computeConfiguration: CohereTranscribeComputeConfiguration
    ) async throws -> CohereTranscribeAsrModels {
        let manifest = try self.loadManifest(from: directory)
        cohereLogger.info(
            "Loading Cohere manifest [modelID=\(manifest.modelID, privacy: .public), sampleRate=\(manifest.sampleRate), decoderMaxLen=\(manifest.decoderMaxLen), hasCachedDecoder=\(manifest.decoderCached != nil), hasCrossKV=\(manifest.crossKVProjector != nil)]"
        )

        async let frontend = self.loadPackage(
            named: manifest.frontend.package,
            from: directory,
            computeUnits: computeConfiguration.frontend
        )
        async let encoder = self.loadPackage(
            named: manifest.encoder.package,
            from: directory,
            computeUnits: computeConfiguration.encoder
        )
        async let decoder = self.loadPackage(
            named: manifest.decoder.package,
            from: directory,
            computeUnits: computeConfiguration.decoder
        )

        let crossKVProjector: MLModel?
        if let crossKVMeta = manifest.crossKVProjector {
            crossKVProjector = try await self.loadPackage(
                named: crossKVMeta.package,
                from: directory,
                computeUnits: computeConfiguration.crossKV
            )
        } else {
            crossKVProjector = nil
        }

        let cachedDecoder: MLModel?
        if let cached = manifest.decoderCached {
            cachedDecoder = try await self.loadPackage(
                named: cached.package,
                from: directory,
                computeUnits: computeConfiguration.decoder
            )
        } else {
            cachedDecoder = nil
        }

        let models = try await CohereTranscribeAsrModels(
            frontend: frontend,
            encoder: encoder,
            decoder: decoder,
            crossKVProjector: crossKVProjector,
            cachedDecoder: cachedDecoder,
            manifest: manifest,
            computeConfiguration: computeConfiguration
        )
        try self.validateLoadedModelContracts(models)
        return models
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

    private static let legacyRequiredFiles = [
        Self.manifestFile,
        "cohere_frontend.mlpackage",
        "cohere_encoder.mlpackage",
        "cohere_decoder_fullseq_masked.mlpackage",
        "cohere_cross_kv_projector.mlpackage",
        "cohere_decoder_cached.mlpackage",
    ]

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
        named packageName: String,
        from directory: URL,
        computeUnits: MLComputeUnits
    ) async throws -> MLModel {
        let packageURL = directory.appendingPathComponent(packageName)
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw CohereTranscribeAsrError.modelNotFound(packageName)
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        configuration.allowLowPrecisionAccumulationOnGPU = true

        let compiledRootDirectory = self.compiledArtifactsDirectory(for: directory)
        let compiledURL = try self.compiledModelURL(
            for: packageURL,
            cacheRoot: compiledRootDirectory,
            computeUnits: computeUnits
        )
        cohereLogger.info(
            "Loading Cohere package \(packageName, privacy: .public) from compiled cache \(compiledURL.path, privacy: .public)"
        )
        return try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
    }

    private static func validateLoadedModelContracts(_ models: CohereTranscribeAsrModels) throws {
        let manifest = models.manifest

        try self.validateRequiredFeatureNames(
            stageName: "frontend",
            availableInputs: models.frontend.modelDescription.inputDescriptionsByName,
            expectedInputs: manifest.frontend.inputs,
            availableOutputs: models.frontend.modelDescription.outputDescriptionsByName,
            expectedOutputs: manifest.frontend.outputs
        )
        try self.validateMultiArrayShape(
            stageName: "frontend",
            featureName: manifest.frontend.inputs[0],
            descriptions: models.frontend.modelDescription.inputDescriptionsByName,
            expectedShape: [1, manifest.maxAudioSamples]
        )
        try self.validateMultiArrayShape(
            stageName: "frontend",
            featureName: manifest.frontend.inputs[1],
            descriptions: models.frontend.modelDescription.inputDescriptionsByName,
            expectedShape: [1]
        )

        try self.validateRequiredFeatureNames(
            stageName: "encoder",
            availableInputs: models.encoder.modelDescription.inputDescriptionsByName,
            expectedInputs: manifest.encoder.inputs,
            availableOutputs: models.encoder.modelDescription.outputDescriptionsByName,
            expectedOutputs: manifest.encoder.outputs
        )

        try self.validateRequiredFeatureNames(
            stageName: "decoder",
            availableInputs: models.decoder.modelDescription.inputDescriptionsByName,
            expectedInputs: manifest.decoder.inputs,
            availableOutputs: models.decoder.modelDescription.outputDescriptionsByName,
            expectedOutputs: manifest.decoder.outputs
        )
        if manifest.decoder.inputs.indices.contains(1) {
            try self.validateMultiArrayShape(
                stageName: "decoder",
                featureName: manifest.decoder.inputs[1],
                descriptions: models.decoder.modelDescription.inputDescriptionsByName,
                expectedShape: [1, manifest.decoderMaxLen]
            )
        }
        if manifest.decoder.inputs.indices.contains(2) {
            try self.validateMultiArrayShape(
                stageName: "decoder",
                featureName: manifest.decoder.inputs[2],
                descriptions: models.decoder.modelDescription.inputDescriptionsByName,
                expectedShape: [1, manifest.decoderMaxLen]
            )
        }

        if let crossKVMeta = manifest.crossKVProjector, let crossKVProjector = models.crossKVProjector {
            try self.validateRequiredFeatureNames(
                stageName: "cross-kv projector",
                availableInputs: crossKVProjector.modelDescription.inputDescriptionsByName,
                expectedInputs: crossKVMeta.inputs,
                availableOutputs: crossKVProjector.modelDescription.outputDescriptionsByName,
                expectedOutputs: crossKVMeta.outputs
            )
        }

        if let cachedMeta = manifest.decoderCached, let cachedDecoder = models.cachedDecoder {
            try self.validateRequiredFeatureNames(
                stageName: "cached decoder",
                availableInputs: cachedDecoder.modelDescription.inputDescriptionsByName,
                expectedInputs: cachedMeta.inputs,
                availableOutputs: cachedDecoder.modelDescription.outputDescriptionsByName,
                expectedOutputs: cachedMeta.outputs
            )
            try self.validateOutputBackingFeature(
                stageName: "cached decoder",
                featureName: cachedMeta.logitsOutput,
                descriptions: cachedDecoder.modelDescription.outputDescriptionsByName
            )
            if let cacheKOutput = cachedMeta.cacheKOutput {
                try self.validateOutputBackingFeature(
                    stageName: "cached decoder",
                    featureName: cacheKOutput,
                    descriptions: cachedDecoder.modelDescription.outputDescriptionsByName
                )
            }
            if let cacheVOutput = cachedMeta.cacheVOutput {
                try self.validateOutputBackingFeature(
                    stageName: "cached decoder",
                    featureName: cacheVOutput,
                    descriptions: cachedDecoder.modelDescription.outputDescriptionsByName
                )
            }
        }
    }

    private static func validateRequiredFeatureNames(
        stageName: String,
        availableInputs: [String: MLFeatureDescription],
        expectedInputs: [String],
        availableOutputs: [String: MLFeatureDescription],
        expectedOutputs: [String]
    ) throws {
        let missingInputs = expectedInputs.filter { availableInputs[$0] == nil }
        if missingInputs.isEmpty == false {
            throw CohereTranscribeAsrError.invalidModelContract(
                "\(stageName) is missing manifest-declared inputs: \(missingInputs.joined(separator: ", "))"
            )
        }

        let missingOutputs = expectedOutputs.filter { availableOutputs[$0] == nil }
        if missingOutputs.isEmpty == false {
            throw CohereTranscribeAsrError.invalidModelContract(
                "\(stageName) is missing manifest-declared outputs: \(missingOutputs.joined(separator: ", "))"
            )
        }
    }

    private static func validateMultiArrayShape(
        stageName: String,
        featureName: String,
        descriptions: [String: MLFeatureDescription],
        expectedShape: [Int]
    ) throws {
        guard let description = descriptions[featureName] else {
            throw CohereTranscribeAsrError.invalidModelContract(
                "\(stageName) feature '\(featureName)' is missing from the model description"
            )
        }
        guard let constraint = description.multiArrayConstraint else {
            throw CohereTranscribeAsrError.invalidModelContract(
                "\(stageName) feature '\(featureName)' is not a constrained MLMultiArray"
            )
        }

        let actualShape = constraint.shape.map(\.intValue)
        guard actualShape == expectedShape else {
            throw CohereTranscribeAsrError.invalidModelContract(
                "\(stageName) feature '\(featureName)' has shape \(actualShape), expected \(expectedShape)"
            )
        }
    }

    private static func validateOutputBackingFeature(
        stageName: String,
        featureName: String,
        descriptions: [String: MLFeatureDescription]
    ) throws {
        guard let description = descriptions[featureName] else {
            throw CohereTranscribeAsrError.invalidModelContract(
                "\(stageName) output '\(featureName)' is missing from the model description"
            )
        }
        guard description.multiArrayConstraint != nil else {
            throw CohereTranscribeAsrError.invalidModelContract(
                "\(stageName) output '\(featureName)' cannot be used with MLMultiArray output backing"
            )
        }
    }

    private static func compiledModelURL(
        for packageURL: URL,
        cacheRoot: URL,
        computeUnits: MLComputeUnits
    ) throws -> URL {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: cacheRoot, withIntermediateDirectories: true)

        let attributes = try fileManager.attributesOfItem(atPath: packageURL.path)
        let modifiedAt = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let baseName = packageURL.deletingPathExtension().lastPathComponent
        let compiledName = "\(baseName)_\(self.computeUnitsKey(computeUnits))_\(Int64(modifiedAt * 1000)).mlmodelc"
        let compiledURL = cacheRoot.appendingPathComponent(compiledName, isDirectory: true)

        if fileManager.fileExists(atPath: compiledURL.path) {
            cohereLogger.debug("Reusing compiled Cohere model at \(compiledURL.path, privacy: .public)")
            return compiledURL
        }

        let lockURL = cacheRoot.appendingPathComponent("\(compiledName).lock", isDirectory: true)
        while true {
            do {
                try fileManager.createDirectory(at: lockURL, withIntermediateDirectories: false)
                break
            } catch {
                if fileManager.fileExists(atPath: lockURL.path) {
                    Thread.sleep(forTimeInterval: 0.1)
                    continue
                }
                throw error
            }
        }
        defer { try? fileManager.removeItem(at: lockURL) }

        if fileManager.fileExists(atPath: compiledURL.path) {
            return compiledURL
        }

        cohereLogger.info("Compiling Cohere package \(packageURL.lastPathComponent, privacy: .public) into \(compiledURL.path, privacy: .public)")
        let tempCompiledURL = try MLModel.compileModel(at: packageURL)
        let stagedURL = cacheRoot.appendingPathComponent("\(compiledName).staging.\(UUID().uuidString)", isDirectory: true)
        try? fileManager.removeItem(at: stagedURL)
        try fileManager.copyItem(at: tempCompiledURL, to: stagedURL)

        if fileManager.fileExists(atPath: compiledURL.path) {
            try? fileManager.removeItem(at: stagedURL)
            try? fileManager.removeItem(at: tempCompiledURL)
            return compiledURL
        }

        try fileManager.moveItem(at: stagedURL, to: compiledURL)
        try? fileManager.removeItem(at: tempCompiledURL)
        cohereLogger.info("Compiled Cohere package ready at \(compiledURL.path, privacy: .public)")
        return compiledURL
    }

    private static func computeUnitsKey(_ computeUnits: MLComputeUnits) -> String {
        switch computeUnits {
        case .cpuOnly:
            return "cpu"
        case .cpuAndGPU:
            return "gpu"
        case .cpuAndNeuralEngine:
            return "ane"
        case .all:
            return "all"
        @unknown default:
            return "unknown"
        }
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
