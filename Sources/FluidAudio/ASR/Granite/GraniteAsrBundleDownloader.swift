import Foundation

enum GraniteAsrBundleDownloader {
    static let repoPath = "FluidInference/granite-speech-4.1-2b-nar-coreml"
    static let folderName = "granite-speech-4.1-2b-nar-coreml"

    private static let packageDirectories: Set<String> = [
        "granite_bpe_greedy_35s.mlpackage",
        "granite_bpe_greedy_60s.mlpackage"
    ]

    private static let requiredRootFiles: Set<String> = [
        "granite_manifest.json",
        "granite_mel_filters_80x257_f32.bin",
        "tokenizer.json"
    ]

    static func download(
        to targetDirectory: URL,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws {
        try FileManager.default.createDirectory(at: targetDirectory, withIntermediateDirectories: true)
        progressHandler?(.init(fractionCompleted: 0.0, phase: .listing))

        let files = try await listFiles(path: "")
        for (index, file) in files.enumerated() {
            try await downloadFile(file.path, to: targetDirectory)
            progressHandler?(
                .init(
                    fractionCompleted: Double(index + 1) / Double(max(files.count, 1)),
                    phase: .downloading(completedFiles: index + 1, totalFiles: files.count)
                )
            )
        }
        try validateBundle(at: targetDirectory)
    }

    private static func listFiles(path: String) async throws -> [RemoteFile] {
        let apiPath = path.isEmpty ? "tree/main" : "tree/main/\(path)"
        let url = try ModelRegistry.apiModels(repoPath, apiPath)
        let (data, _) = try await DownloadUtils.fetchWithAuth(from: url)
        let items = try JSONDecoder().decode([RemoteItem].self, from: data)

        var files: [RemoteFile] = []
        for item in items {
            if item.type == "directory", shouldVisitDirectory(item.path) {
                files.append(contentsOf: try await listFiles(path: item.path))
            } else if item.type == "file", shouldDownloadFile(item.path) {
                files.append(RemoteFile(path: item.path))
            }
        }
        return files
    }

    private static func shouldVisitDirectory(_ path: String) -> Bool {
        packageDirectories.contains { package in
            path == package || package.hasPrefix("\(path)/") || path.hasPrefix("\(package)/")
        }
    }

    private static func shouldDownloadFile(_ path: String) -> Bool {
        if packageDirectories.contains(where: { path.hasPrefix("\($0)/") }) {
            return true
        }

        guard !path.contains("/") else { return false }
        return requiredRootFiles.contains(path)
            || path.hasSuffix(".json")
            || path.hasSuffix(".txt")
            || path.hasSuffix(".model")
            || path.hasSuffix(".bin")
    }

    private static func downloadFile(_ path: String, to targetDirectory: URL) async throws {
        let destination = targetDirectory.appendingPathComponent(path)
        if FileManager.default.fileExists(atPath: destination.path) {
            return
        }

        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let encodedPath = path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? path
        let url = try ModelRegistry.resolveModel(repoPath, encodedPath)
        let (temporaryURL, response) = try await DownloadUtils.sharedSession.download(from: url)
        guard let httpResponse = response as? HTTPURLResponse, (200 ..< 300).contains(httpResponse.statusCode) else {
            throw GraniteAsrError.modelNotFound(path)
        }
        try FileManager.default.moveItem(at: temporaryURL, to: destination)
    }

    private static func validateBundle(at directory: URL) throws {
        for file in requiredRootFiles {
            let url = directory.appendingPathComponent(file)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw GraniteAsrError.modelNotFound(file)
            }
        }

        for package in packageDirectories {
            let url = directory.appendingPathComponent(package)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw GraniteAsrError.modelNotFound(package)
            }
        }
    }
}

private struct RemoteItem: Decodable {
    let path: String
    let type: String
}

private struct RemoteFile {
    let path: String
}
