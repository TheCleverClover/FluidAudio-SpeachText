import Foundation

public struct CommonVoiceSample: Equatable, Sendable {
    public let sampleId: String
    public let audioPath: URL
    public let transcript: String
    public let relativePath: String
    public let metadata: [String: String]

    public init(
        sampleId: String,
        audioPath: URL,
        transcript: String,
        relativePath: String,
        metadata: [String: String] = [:]
    ) {
        self.sampleId = sampleId
        self.audioPath = audioPath
        self.transcript = transcript
        self.relativePath = relativePath
        self.metadata = metadata
    }
}

public enum CommonVoiceManifest {
    public static func loadSamples(
        datasetDirectory: URL,
        split: String = "test",
        language: String? = nil,
        variant: String? = nil
    ) throws -> [CommonVoiceSample] {
        let manifestURL = try findManifest(
            datasetDirectory: datasetDirectory,
            split: split,
            language: language
        )
        return try parseManifest(
            manifestURL,
            variant: variant
        )
    }

    static func parseManifest(
        _ manifestURL: URL,
        variant: String? = nil
    ) throws -> [CommonVoiceSample] {
        let content = try String(contentsOf: manifestURL, encoding: .utf8)
        var lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
        guard !lines.isEmpty else {
            throw ASRError.processingFailed("Common Voice manifest is empty: \(manifestURL.path)")
        }

        let headerLine = lines.removeFirst()
        let headers = splitTSVLine(headerLine).map { header in
            header.trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: "\u{feff}", with: "")
        }
        let normalizedHeaders = headers.map { $0.lowercased() }

        guard let pathIndex = firstColumn(named: ["path", "audio", "audio_path"], in: normalizedHeaders) else {
            throw ASRError.processingFailed("Common Voice manifest missing path column: \(manifestURL.path)")
        }
        guard
            let sentenceIndex = firstColumn(
                named: ["sentence", "text", "transcript", "transcription"],
                in: normalizedHeaders)
        else {
            throw ASRError.processingFailed("Common Voice manifest missing sentence/text column: \(manifestURL.path)")
        }

        let variantColumns = ["variant", "locale", "accent", "accents"].compactMap {
            firstColumn(named: [$0], in: normalizedHeaders)
        }
        let variantFilter = variant?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        var samples: [CommonVoiceSample] = []
        samples.reserveCapacity(lines.count)

        for (lineIndex, line) in lines.enumerated() {
            let fields = paddedFields(splitTSVLine(line), count: headers.count)
            let relativePath = fields[pathIndex].trimmingCharacters(in: .whitespacesAndNewlines)
            let transcript = fields[sentenceIndex].trimmingCharacters(in: .whitespacesAndNewlines)

            guard !relativePath.isEmpty, !transcript.isEmpty else {
                continue
            }

            if let variantFilter, !variantFilter.isEmpty {
                let matchesVariant = variantColumns.contains { column in
                    fields[column].lowercased().contains(variantFilter)
                }
                if !matchesVariant {
                    continue
                }
            }

            var metadata: [String: String] = [:]
            metadata.reserveCapacity(headers.count)
            for (index, header) in headers.enumerated() where index < fields.count {
                metadata[header] = fields[index]
            }

            let audioURL = resolveAudioURL(relativePath, manifestURL: manifestURL)
            let sampleId =
                URL(fileURLWithPath: relativePath).deletingPathExtension().lastPathComponent.nonEmpty
                ?? "common_voice_\(lineIndex + 1)"

            samples.append(
                CommonVoiceSample(
                    sampleId: sampleId,
                    audioPath: audioURL,
                    transcript: transcript,
                    relativePath: relativePath,
                    metadata: metadata
                )
            )
        }

        guard !samples.isEmpty else {
            throw ASRError.processingFailed("Common Voice manifest has no usable samples: \(manifestURL.path)")
        }
        return samples
    }

    private static func findManifest(
        datasetDirectory: URL,
        split: String,
        language: String?
    ) throws -> URL {
        let fileManager = FileManager.default
        let manifestName = "\(split).tsv"
        var candidates: [URL] = []

        candidates.append(datasetDirectory.appendingPathComponent(manifestName))
        if let language, !language.isEmpty {
            candidates.append(
                datasetDirectory
                    .appendingPathComponent(language)
                    .appendingPathComponent(manifestName)
            )
        }

        for candidate in candidates where fileManager.fileExists(atPath: candidate.path) {
            return candidate
        }

        guard
            let enumerator = fileManager.enumerator(
                at: datasetDirectory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
        else {
            throw ASRError.processingFailed("Unable to scan Common Voice directory: \(datasetDirectory.path)")
        }

        var recursiveMatches: [URL] = []
        while let url = enumerator.nextObject() as? URL {
            if url.lastPathComponent == manifestName {
                recursiveMatches.append(url)
            }
        }

        if let language, !language.isEmpty {
            let languageMatches = recursiveMatches.filter { url in
                url.pathComponents.contains(language)
            }
            if let match = languageMatches.sorted(by: { $0.path.count < $1.path.count }).first {
                return match
            }
        }

        if let match = recursiveMatches.sorted(by: { $0.path.count < $1.path.count }).first {
            return match
        }

        throw ASRError.processingFailed(
            "No Common Voice \(manifestName) found under \(datasetDirectory.path)"
        )
    }

    private static func splitTSVLine(_ line: String) -> [String] {
        line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
    }

    private static func paddedFields(_ fields: [String], count: Int) -> [String] {
        if fields.count >= count {
            return fields
        }
        return fields + Array(repeating: "", count: count - fields.count)
    }

    private static func firstColumn(named names: [String], in headers: [String]) -> Int? {
        for name in names {
            if let index = headers.firstIndex(of: name) {
                return index
            }
        }
        return nil
    }

    private static func resolveAudioURL(_ path: String, manifestURL: URL) -> URL {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path)
        }

        let manifestDirectory = manifestURL.deletingLastPathComponent()
        if path.contains("/") {
            return manifestDirectory.appendingPathComponent(path)
        }

        let clipsURL = manifestDirectory.appendingPathComponent("clips").appendingPathComponent(path)
        if FileManager.default.fileExists(atPath: clipsURL.path) {
            return clipsURL
        }
        return manifestDirectory.appendingPathComponent(path)
    }
}

extension String {
    fileprivate var nonEmpty: String? {
        isEmpty ? nil : self
    }
}
