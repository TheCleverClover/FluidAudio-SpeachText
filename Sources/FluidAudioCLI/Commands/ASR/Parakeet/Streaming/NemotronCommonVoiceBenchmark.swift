#if os(macOS)
import AVFoundation
@preconcurrency import CoreML
import FluidAudio
import Foundation

public final class NemotronCommonVoiceBenchmark {
    private let logger = AppLogger(category: "NemotronCommonVoiceBenchmark")

    public struct Config {
        var datasetDir: URL?
        var language: String = "pt"
        var split: String = "test"
        var variant: String?
        var maxFiles: Int?
        var modelDir: URL?
        var manifest: URL?
        var targetLang: String?
        var computeUnits: String = "cpuAndNeuralEngine"
        var allowLowPrecisionGPU: Bool = true
        var output: URL?

        public init() {}
    }

    private struct BenchmarkSummary: Codable {
        let datasetDir: String
        let language: String
        let split: String
        let variant: String?
        let targetLang: String?
        let modelDir: String
        let filesProcessed: Int
        let filesSkipped: Int
        let totalWords: Int
        let totalErrors: Int
        let insertions: Int
        let deletions: Int
        let substitutions: Int
        let wer: Double
        let audioDuration: Double
        let processingTime: Double
        let rtfx: Double
    }

    private struct FileResult {
        let hypothesis: String
        let metrics: (
            wer: Double,
            cer: Double,
            insertions: Int,
            deletions: Int,
            substitutions: Int,
            totalWords: Int,
            totalCharacters: Int
        )
        let audioDuration: Double
        let processingTime: Double
    }

    private struct BenchmarkSample {
        let sampleId: String
        let audioPath: URL
        let transcript: String
        let relativePath: String
    }

    private struct JSONManifestRecord: Decodable {
        let audio_filepath: String
        let text: String
        let duration: Double?
        let target_lang: String?
    }

    private let config: Config

    public init(config: Config = Config()) {
        self.config = config
    }

    public static func run(arguments: [String]) async {
        let logger = AppLogger(category: "NemotronCommonVoiceBenchmark")
        var config = Config()

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]

            switch arg {
            case "--dataset-dir", "-d":
                i += 1
                if i < arguments.count {
                    config.datasetDir = URL(fileURLWithPath: arguments[i])
                }
            case "--language", "-l":
                i += 1
                if i < arguments.count {
                    config.language = arguments[i]
                }
            case "--split", "-s":
                i += 1
                if i < arguments.count {
                    config.split = arguments[i]
                }
            case "--variant":
                i += 1
                if i < arguments.count {
                    config.variant = arguments[i]
                }
            case "--max-files", "-n":
                i += 1
                if i < arguments.count, let maxFiles = Int(arguments[i]) {
                    config.maxFiles = maxFiles
                }
            case "--model-dir", "-m":
                i += 1
                if i < arguments.count {
                    config.modelDir = URL(fileURLWithPath: arguments[i])
                }
            case "--manifest":
                i += 1
                if i < arguments.count {
                    config.manifest = URL(fileURLWithPath: arguments[i])
                }
            case "--target-lang":
                i += 1
                if i < arguments.count {
                    config.targetLang = arguments[i]
                }
            case "--compute-units":
                i += 1
                if i < arguments.count {
                    config.computeUnits = arguments[i]
                }
            case "--disable-low-precision-gpu":
                config.allowLowPrecisionGPU = false
            case "--output", "-o":
                i += 1
                if i < arguments.count {
                    config.output = URL(fileURLWithPath: arguments[i])
                }
            case "--help", "-h":
                printUsage()
                return
            default:
                logger.warning("Unknown argument: \(arg)")
            }
            i += 1
        }

        let benchmark = NemotronCommonVoiceBenchmark(config: config)
        await benchmark.run()
    }

    private static func printUsage() {
        print(
            """
            Nemotron Common Voice Benchmark

            Usage: fluidaudio nemotron-commonvoice-benchmark [options]

            Options:
                --dataset-dir, -d <path>  Extracted Common Voice directory
                --language, -l <locale>   Common Voice locale to find (default: pt)
                --split, -s <name>        TSV split name (default: test)
                --variant <value>         Optional variant/accent/locale filter
                --max-files, -n <count>   Maximum files to process
                --model-dir, -m <path>    Path to Nemotron CoreML models
                --manifest <path>         NeMo-style JSONL manifest with audio_filepath/text
                --target-lang <lang>      Runtime prompt language (default: language)
                --compute-units <units>   cpuAndNeuralEngine, all, cpuAndGPU, cpuOnly (default: cpuAndNeuralEngine)
                --disable-low-precision-gpu
                --output, -o <path>       JSON output path
                --help, -h                Show this help

            Example:
                fluidaudio nemotron-commonvoice-benchmark -d ~/Downloads/cv-corpus --language pt --target-lang pt
            """
        )
    }

    public func run() async {
        do {
            let datasetDir = try resolveDatasetDirectory()
            let modelDir = try resolveModelDirectory()
            let samples = try loadSamples(datasetDir: datasetDir)
            let filesToProcess = Array(samples.prefix(config.maxFiles ?? samples.count))

            guard !filesToProcess.isEmpty else {
                throw ASRError.processingFailed("No Common Voice samples selected")
            }

            if let manifest = config.manifest {
                print("Manifest \(manifest.path): \(filesToProcess.count) files")
            } else {
                print("Common Voice \(config.language)/\(config.split): \(filesToProcess.count) files")
            }
            print("Model: \(modelDir.path)")

            let mlConfig = try makeModelConfiguration()
            let manager = NemotronStreamingAsrManager(configuration: mlConfig)
            try await manager.loadModels(modelDir: modelDir)
            print("Compute units: \(config.computeUnits)")
            print("Low precision GPU accumulation: \(config.allowLowPrecisionGPU)")

            let runtimePrompt = await manager.config.runtimePrompt
            var activePromptLanguage: String?
            if let targetLang = config.targetLang {
                try await manager.setTargetLanguage(targetLang)
                activePromptLanguage = targetLang
            } else if runtimePrompt {
                let targetLang = defaultPromptLanguage(for: config.language)
                try await manager.setTargetLanguage(targetLang)
                activePromptLanguage = targetLang
            }
            if let activePromptLanguage {
                print("Runtime prompt language: \(activePromptLanguage)")
            }

            var filesProcessed = 0
            var filesSkipped = 0
            var totalWords = 0
            var totalErrors = 0
            var totalInsertions = 0
            var totalDeletions = 0
            var totalSubstitutions = 0
            var totalAudioDuration: Double = 0
            var totalProcessingTime: Double = 0

            let progressInterval = max(1, filesToProcess.count / 20)

            for (index, sample) in filesToProcess.enumerated() {
                guard FileManager.default.fileExists(atPath: sample.audioPath.path) else {
                    filesSkipped += 1
                    logger.warning("Missing audio: \(sample.audioPath.path)")
                    continue
                }

                do {
                    let result = try await processSample(manager: manager, sample: sample)
                    let sampleErrors =
                        result.metrics.insertions + result.metrics.deletions + result.metrics.substitutions

                    totalWords += result.metrics.totalWords
                    totalErrors += sampleErrors
                    totalInsertions += result.metrics.insertions
                    totalDeletions += result.metrics.deletions
                    totalSubstitutions += result.metrics.substitutions
                    totalAudioDuration += result.audioDuration
                    totalProcessingTime += result.processingTime
                    filesProcessed += 1

                    if filesProcessed == 1 || filesProcessed % progressInterval == 0 || index == filesToProcess.count - 1
                    {
                        let runningWER =
                            totalWords > 0 ? Double(totalErrors) / Double(totalWords) * 100.0 : 0.0
                        let runningRTFx =
                            totalProcessingTime > 0 ? totalAudioDuration / totalProcessingTime : 0.0
                        print(
                            "[\(index + 1)/\(filesToProcess.count)] WER \(String(format: "%.2f", runningWER))%, RTFx \(String(format: "%.1f", runningRTFx))x"
                        )
                    }

                    await manager.reset()
                } catch {
                    filesSkipped += 1
                    await manager.reset()
                    logger.warning("Failed \(sample.relativePath): \(error.localizedDescription)")
                }
            }

            guard filesProcessed > 0 else {
                throw ASRError.processingFailed("No Common Voice files were processed")
            }
            guard totalWords > 0 else {
                throw ASRError.processingFailed("No reference words found in processed Common Voice samples")
            }
            guard totalAudioDuration > 0, totalProcessingTime > 0 else {
                throw ASRError.processingFailed("Benchmark timing is invalid")
            }

            let wer = Double(totalErrors) / Double(totalWords) * 100.0
            let rtfx = totalAudioDuration / totalProcessingTime

            let summary = BenchmarkSummary(
                datasetDir: datasetDir.path,
                language: config.language,
                split: config.split,
                variant: config.variant,
                targetLang: activePromptLanguage,
                modelDir: modelDir.path,
                filesProcessed: filesProcessed,
                filesSkipped: filesSkipped,
                totalWords: totalWords,
                totalErrors: totalErrors,
                insertions: totalInsertions,
                deletions: totalDeletions,
                substitutions: totalSubstitutions,
                wer: wer,
                audioDuration: totalAudioDuration,
                processingTime: totalProcessingTime,
                rtfx: rtfx
            )

            let outputURL = config.output ?? defaultOutputURL(language: config.language, split: config.split)
            try save(summary, to: outputURL)

            print("")
            print("SUMMARY")
            print("Files processed: \(filesProcessed)")
            print("Files skipped:   \(filesSkipped)")
            print("Total words:     \(totalWords)")
            print("WER:             \(String(format: "%.2f", wer))%")
            print("Audio duration:  \(String(format: "%.1f", totalAudioDuration))s")
            print("Processing time: \(String(format: "%.1f", totalProcessingTime))s")
            print("RTFx:            \(String(format: "%.1f", rtfx))x")
            print("Results saved to \(outputURL.path)")
        } catch {
            logger.error("Common Voice benchmark failed: \(error.localizedDescription)")
        }
    }

    private func processSample(
        manager: NemotronStreamingAsrManager,
        sample: BenchmarkSample
    ) async throws -> FileResult {
        let audioFile = try AVAudioFile(forReading: sample.audioPath)
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: audioFile.processingFormat,
                frameCapacity: AVAudioFrameCount(audioFile.length)
            )
        else {
            throw ASRError.processingFailed("Failed to create audio buffer for \(sample.audioPath.lastPathComponent)")
        }
        try audioFile.read(into: buffer)

        let audioDuration = Double(audioFile.length) / audioFile.processingFormat.sampleRate
        let startTime = Date()
        let hypothesis = try await manager.transcribe(audioBuffer: buffer)
        let processingTime = Date().timeIntervalSince(startTime)
        let metrics = WERCalculator.calculateWERAndCER(
            hypothesis: hypothesis,
            reference: sample.transcript
        )

        return FileResult(
            hypothesis: hypothesis,
            metrics: metrics,
            audioDuration: audioDuration,
            processingTime: processingTime
        )
    }

    private func loadSamples(datasetDir: URL) throws -> [BenchmarkSample] {
        if let manifest = config.manifest {
            return try loadJSONManifest(manifest)
        }

        return try CommonVoiceManifest.loadSamples(
            datasetDirectory: datasetDir,
            split: config.split,
            language: config.language,
            variant: config.variant
        ).map { sample in
            BenchmarkSample(
                sampleId: sample.sampleId,
                audioPath: sample.audioPath,
                transcript: sample.transcript,
                relativePath: sample.relativePath
            )
        }
    }

    private func loadJSONManifest(_ manifestURL: URL) throws -> [BenchmarkSample] {
        let content = try String(contentsOf: manifestURL, encoding: .utf8)
        let lines = content.split(whereSeparator: \.isNewline)
        let decoder = JSONDecoder()
        let manifestDir = manifestURL.deletingLastPathComponent()

        var samples: [BenchmarkSample] = []
        samples.reserveCapacity(lines.count)

        for (index, line) in lines.enumerated() {
            let data = Data(line.utf8)
            let record = try decoder.decode(JSONManifestRecord.self, from: data)
            let audioURL: URL
            if record.audio_filepath.hasPrefix("/") {
                audioURL = URL(fileURLWithPath: record.audio_filepath)
            } else {
                audioURL = manifestDir.appendingPathComponent(record.audio_filepath)
            }
            let sampleId = audioURL.deletingPathExtension().lastPathComponent
            samples.append(
                BenchmarkSample(
                    sampleId: sampleId.isEmpty ? "manifest_\(index + 1)" : sampleId,
                    audioPath: audioURL,
                    transcript: record.text,
                    relativePath: record.audio_filepath
                )
            )
        }

        guard !samples.isEmpty else {
            throw ASRError.processingFailed("Manifest has no usable samples: \(manifestURL.path)")
        }
        return samples
    }

    private func resolveDatasetDirectory() throws -> URL {
        if let manifest = config.manifest {
            return manifest.deletingLastPathComponent()
        }
        if let datasetDir = config.datasetDir {
            return datasetDir
        }

        let defaultDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/fluidaudio/datasets/common_voice")
        guard FileManager.default.fileExists(atPath: defaultDir.path) else {
            throw ASRError.processingFailed(
                "Pass --dataset-dir with an extracted Mozilla Common Voice corpus"
            )
        }
        return defaultDir
    }

    private func resolveModelDirectory() throws -> URL {
        guard let modelDir = config.modelDir else {
            throw ASRError.processingFailed("Pass --model-dir with the Nemotron CoreML bundle")
        }
        return modelDir
    }

    private func save(_ summary: BenchmarkSummary, to outputURL: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(summary)
        let directory = outputURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try data.write(to: outputURL)
    }

    private func makeModelConfiguration() throws -> MLModelConfiguration {
        let mlConfig = MLModelConfiguration()
        mlConfig.allowLowPrecisionAccumulationOnGPU = config.allowLowPrecisionGPU
        switch config.computeUnits.lowercased() {
        case "all":
            mlConfig.computeUnits = .all
        case "cpuandneuralengine", "cpu-ne", "ane":
            mlConfig.computeUnits = .cpuAndNeuralEngine
        case "cpuandgpu", "cpu-gpu", "gpu":
            mlConfig.computeUnits = .cpuAndGPU
        case "cpuonly", "cpu":
            mlConfig.computeUnits = .cpuOnly
        default:
            throw ASRError.processingFailed("Unknown compute units: \(config.computeUnits)")
        }
        return mlConfig
    }

    private func defaultOutputURL(language: String, split: String) -> URL {
        let safeLanguage = language.replacingOccurrences(of: "/", with: "_")
        let safeSplit = split.replacingOccurrences(of: "/", with: "_")
        return URL(fileURLWithPath: "/tmp/nemotron_commonvoice_\(safeLanguage)_\(safeSplit)_benchmark.json")
    }

    private func defaultPromptLanguage(for language: String) -> String {
        switch language.lowercased() {
        case "en", "en-us":
            return "en-US"
        case "pt", "pt-pt":
            return "pt"
        case "pt-br":
            return "pt-BR"
        case "es", "fr", "it":
            return language.lowercased()
        default:
            return language
        }
    }
}
#endif
