@preconcurrency import CoreML
import Foundation
import OSLog
@preconcurrency import Tokenizers

private let graniteModelLogger = Logger(subsystem: "FluidAudio", category: "GraniteAsrModels")
private typealias HFTokenizerProtocol = Tokenizers.Tokenizer

public enum GraniteAsrError: LocalizedError {
    case manifestMissing(URL)
    case invalidManifest(URL, Error)
    case modelNotFound(String)
    case tokenizerNotFound(URL)
    case tokenizerFailed(Error)
    case invalidOutput(String)
    case invalidAudio(String)

    public var errorDescription: String? {
        switch self {
        case let .manifestMissing(url):
            return "Granite manifest not found at \(url.path)"
        case let .invalidManifest(url, error):
            return "Failed to decode Granite manifest at \(url.path): \(error.localizedDescription)"
        case let .modelNotFound(name):
            return "Required Granite CoreML model not found: \(name)"
        case let .tokenizerNotFound(url):
            return "Granite tokenizer not found at \(url.path)"
        case let .tokenizerFailed(error):
            return "Failed to load Granite tokenizer: \(error.localizedDescription)"
        case let .invalidOutput(message):
            return "Invalid Granite model output: \(message)"
        case let .invalidAudio(message):
            return "Invalid Granite audio: \(message)"
        }
    }
}

public struct GraniteWindowMeta: Codable, Sendable {
    public let seconds: Int
    public let samples: Int
    public let frames: Int
    public let bpeFrames: Int
    public let package: String
    public let inputs: [String]
    public let outputs: [String]

    private enum CodingKeys: String, CodingKey {
        case seconds
        case samples
        case frames
        case bpeFrames = "bpe_frames"
        case package
        case inputs
        case outputs
    }
}

public struct GraniteAsrManifest: Codable, Sendable {
    public let modelID: String
    public let sampleRate: Int
    public let nFFT: Int
    public let winLength: Int
    public let hopLength: Int
    public let nMels: Int
    public let featureFramesPerSecond: Int
    public let bpePoolingWindow: Int
    public let defaultWindowSeconds: Int
    public let defaultOverlapSeconds: Double
    public let speedWindowSeconds: Int
    public let speedOverlapSeconds: Double
    public let melFilters: String
    public let tokenizer: String
    public let windows: [String: GraniteWindowMeta]

    private enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case sampleRate = "sample_rate"
        case nFFT = "n_fft"
        case winLength = "win_length"
        case hopLength = "hop_length"
        case nMels = "n_mels"
        case featureFramesPerSecond = "feature_frames_per_second"
        case bpePoolingWindow = "bpe_pooling_window"
        case defaultWindowSeconds = "default_window_seconds"
        case defaultOverlapSeconds = "default_overlap_seconds"
        case speedWindowSeconds = "speed_window_seconds"
        case speedOverlapSeconds = "speed_overlap_seconds"
        case melFilters = "mel_filters"
        case tokenizer
        case windows
    }
}

public final class GraniteTokenizer {
    private let tokenizer: any HFTokenizerProtocol

    public init(modelDirectory: URL) async throws {
        let tokenizerURL = modelDirectory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw GraniteAsrError.tokenizerNotFound(tokenizerURL)
        }

        do {
            tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)
        } catch {
            throw GraniteAsrError.tokenizerFailed(error)
        }
    }

    public func decode(_ tokenIDs: [Int]) -> String {
        tokenizer.decode(tokens: tokenIDs, skipSpecialTokens: true)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }
}

@available(macOS 14, iOS 17, *)
public struct GraniteAsrModels {
    public static let manifestFile = "granite_manifest.json"

    public let balancedModel: MLModel
    public let speedModel: MLModel?
    public let manifest: GraniteAsrManifest
    public let tokenizer: GraniteTokenizer
    public let modelDirectory: URL

    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU,
        loadSpeedModel: Bool = false
    ) async throws -> GraniteAsrModels {
        let manifest = try loadManifest(from: directory)
        let tokenizer = try await GraniteTokenizer(modelDirectory: directory)
        let balancedKey = "\(manifest.defaultWindowSeconds)s"
        guard let balancedMeta = manifest.windows[balancedKey] else {
            throw GraniteAsrError.modelNotFound("window \(balancedKey)")
        }

        let loadedBalanced = try await loadPackage(
            named: balancedMeta.package,
            from: directory,
            computeUnits: computeUnits
        )

        let speedModel: MLModel?
        let speedKey = "\(manifest.speedWindowSeconds)s"
        if loadSpeedModel, speedKey != balancedKey, let speedMeta = manifest.windows[speedKey] {
            speedModel = try await loadPackage(
                named: speedMeta.package,
                from: directory,
                computeUnits: computeUnits
            )
        } else {
            speedModel = nil
        }

        let path = directory.path
        graniteModelLogger.info("Loaded Granite NAR CoreML models from \(path, privacy: .public)")
        let computeKey = computeUnits.rawValue
        graniteModelLogger.info("Granite balanced window: \(balancedKey, privacy: .public)")
        graniteModelLogger.info("Granite speed window: \(speedKey, privacy: .public)")
        graniteModelLogger.info("Granite compute units: \(computeKey, privacy: .public)")
        return GraniteAsrModels(
            balancedModel: loadedBalanced,
            speedModel: speedModel,
            manifest: manifest,
            tokenizer: tokenizer,
            modelDirectory: directory
        )
    }

    public static func loadWindowModel(
        _ meta: GraniteWindowMeta,
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> MLModel {
        try await loadPackage(
            named: meta.package,
            from: directory,
            computeUnits: computeUnits
        )
    }

    public static func defaultCacheDirectory() -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return appSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent(GraniteAsrBundleDownloader.folderName, isDirectory: true)
    }

    @discardableResult
    public static func download(
        to directory: URL? = nil,
        force: Bool = false,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory()
        if !force, modelsExist(at: targetDir) {
            graniteModelLogger.info("Granite NAR CoreML bundle already present at \(targetDir.path, privacy: .public)")
            return targetDir
        }

        if force, FileManager.default.fileExists(atPath: targetDir.path) {
            try FileManager.default.removeItem(at: targetDir)
        }

        try await GraniteAsrBundleDownloader.download(to: targetDir, progressHandler: progressHandler)
        return targetDir
    }

    public static func downloadAndLoad(
        to directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndGPU,
        loadSpeedModel: Bool = false,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> GraniteAsrModels {
        let targetDir = try await download(to: directory, progressHandler: progressHandler)
        return try await load(from: targetDir, computeUnits: computeUnits, loadSpeedModel: loadSpeedModel)
    }

    public static func modelsExist(at directory: URL) -> Bool {
        guard let manifest = try? loadManifest(from: directory) else {
            return false
        }
        let fileManager = FileManager.default
        let tokenizerPath = directory.appendingPathComponent(manifest.tokenizer)
        let filtersPath = directory.appendingPathComponent(manifest.melFilters)
        guard fileManager.fileExists(atPath: tokenizerPath.path),
              fileManager.fileExists(atPath: filtersPath.path)
        else {
            return false
        }
        return manifest.windows.values.allSatisfy { meta in
            fileManager.fileExists(atPath: directory.appendingPathComponent(meta.package).path)
        }
    }

    private static func loadManifest(from directory: URL) throws -> GraniteAsrManifest {
        let url = directory.appendingPathComponent(Self.manifestFile)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GraniteAsrError.manifestMissing(url)
        }

        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(GraniteAsrManifest.self, from: data)
        } catch {
            throw GraniteAsrError.invalidManifest(url, error)
        }
    }

    private static func loadPackage(
        named packageName: String,
        from directory: URL,
        computeUnits: MLComputeUnits
    ) async throws -> MLModel {
        let packageURL = directory.appendingPathComponent(packageName)
        let compiledSibling = directory.appendingPathComponent(
            packageURL.deletingPathExtension().lastPathComponent + ".mlmodelc",
            isDirectory: true
        )

        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiledSibling.path) {
            modelURL = compiledSibling
        } else if FileManager.default.fileExists(atPath: packageURL.path) {
            modelURL = try compiledModelURL(for: packageURL, sourceDirectory: directory, computeUnits: computeUnits)
        } else {
            throw GraniteAsrError.modelNotFound(packageName)
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        configuration.allowLowPrecisionAccumulationOnGPU = true
        return try await MLModel.load(contentsOf: modelURL, configuration: configuration)
    }

    private static func compiledModelURL(
        for packageURL: URL,
        sourceDirectory: URL,
        computeUnits: MLComputeUnits
    ) throws -> URL {
        let cacheRoot = compiledArtifactsDirectory(for: sourceDirectory)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let attributes = try FileManager.default.attributesOfItem(atPath: packageURL.path)
        let modifiedAt = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let baseName = packageURL.deletingPathExtension().lastPathComponent
        let compiledName = "\(baseName)_\(computeUnitsKey(computeUnits))_\(Int64(modifiedAt * 1000)).mlmodelc"
        let compiledURL = cacheRoot.appendingPathComponent(compiledName, isDirectory: true)

        if FileManager.default.fileExists(atPath: compiledURL.path) {
            return compiledURL
        }

        let tempCompiledURL = try MLModel.compileModel(at: packageURL)
        try? FileManager.default.removeItem(at: compiledURL)
        try FileManager.default.copyItem(at: tempCompiledURL, to: compiledURL)
        try? FileManager.default.removeItem(at: tempCompiledURL)
        return compiledURL
    }

    private static func compiledArtifactsDirectory(for sourceDirectory: URL) -> URL {
        let cachesDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? sourceDirectory.deletingLastPathComponent()
        return cachesDirectory
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("CompiledGraniteModels", isDirectory: true)
            .appendingPathComponent(stableCompiledDirectoryName(for: sourceDirectory), isDirectory: true)
    }

    private static func computeUnitsKey(_ computeUnits: MLComputeUnits) -> String {
        switch computeUnits {
        case .cpuOnly: return "cpu"
        case .cpuAndGPU: return "gpu"
        case .cpuAndNeuralEngine: return "ane"
        case .all: return "all"
        @unknown default: return "unknown"
        }
    }

    private static func stableCompiledDirectoryName(for sourceDirectory: URL) -> String {
        let path = sourceDirectory.standardizedFileURL.path
        let folderName = sourceDirectory.lastPathComponent.replacingOccurrences(of: " ", with: "_")
        return "\(folderName)-\(fnv1a64(path))"
    }

    private static func fnv1a64(_ string: String) -> String {
        var hash: UInt64 = 0xCBF2_9CE4_8422_2325
        let prime: UInt64 = 0x100_0000_01B3
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash &*= prime
        }
        return String(hash, radix: 16, uppercase: false)
    }
}
