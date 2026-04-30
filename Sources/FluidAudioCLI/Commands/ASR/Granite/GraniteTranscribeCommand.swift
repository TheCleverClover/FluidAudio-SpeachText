#if os(macOS)
@preconcurrency import CoreML
import FluidAudio
import Foundation

private struct GraniteTranscribeOptions {
    let audioFile: String
    var modelDir: String?
    var referenceFile: String?
    var mode: GraniteAsrMode = .balanced
    var computeUnits: MLComputeUnits = .cpuAndGPU
}

enum GraniteTranscribeCommand {
    private static let logger = AppLogger(category: "GraniteTranscribe")

    static func run(arguments: [String]) async {
        guard let options = parseOptions(arguments) else {
            return
        }

        await transcribe(options: options)
    }

    private static func parseOptions(_ arguments: [String]) -> GraniteTranscribeOptions? {
        guard !arguments.isEmpty else {
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        var options = GraniteTranscribeOptions(audioFile: arguments[0])
        var index = 1
        while index < arguments.count {
            switch arguments[index] {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--model-dir":
                options.modelDir = nextValue(arguments, at: &index, option: "--model-dir")
            case "--reference":
                options.referenceFile = nextValue(arguments, at: &index, option: "--reference")
            case "--mode":
                guard let value = nextValue(arguments, at: &index, option: "--mode"),
                    let parsed = parseMode(value)
                else {
                    logger.error("Unknown Granite mode")
                    exit(1)
                }
                options.mode = parsed
            case "--compute-units":
                guard let value = nextValue(arguments, at: &index, option: "--compute-units"),
                    let parsed = parseComputeUnits(value)
                else {
                    logger.error("Unknown compute units")
                    exit(1)
                }
                options.computeUnits = parsed
            default:
                logger.warning("Unknown option: \(arguments[index])")
            }
            index += 1
        }
        return options
    }

    private static func nextValue(_ arguments: [String], at index: inout Int, option: String) -> String? {
        guard index + 1 < arguments.count else {
            logger.error("Missing value for \(option)")
            return nil
        }
        index += 1
        return arguments[index]
    }

    private static func transcribe(options: GraniteTranscribeOptions) async {
        guard #available(macOS 14, iOS 17, *) else {
            logger.error("Granite NAR requires macOS 14 or later")
            return
        }

        do {
            let defaultDir = GraniteAsrModels.defaultCacheDirectory().path
            let directory = URL(fileURLWithPath: options.modelDir ?? defaultDir)
            guard GraniteAsrModels.modelsExist(at: directory) else {
                logger.error("Granite CoreML bundle missing at \(directory.path)")
                logger.error("Use --model-dir /path/to/fluid_audio_granite_nar")
                return
            }

            logger.info("Loading Granite NAR bundle from: \(directory.path)")
            let manager = GraniteAsrManager()
            try await manager.loadModels(from: directory, computeUnits: options.computeUnits)

            let samples = try AudioConverter().resampleAudioFile(path: options.audioFile)
            let duration = Double(samples.count) / 16_000.0
            logger.info("Audio: \(String(format: "%.2f", duration))s, \(samples.count) samples at 16kHz")

            let result = try await manager.transcribeDetailed(audioSamples: samples, mode: options.mode)
            let reference = try options.referenceFile.map { path in
                try String(contentsOfFile: path, encoding: .utf8)
            }
            printResult(result, mode: options.mode, reference: reference)
        } catch {
            logger.error("Granite transcription failed: \(error)")
        }
    }

    private static func printResult(_ result: GraniteTranscriptionResult, mode: GraniteAsrMode, reference: String?) {
        let overlap = String(format: "%.1f", result.overlapSeconds)
        let audioDuration = String(format: "%.2f", result.durationSeconds)
        let elapsed = String(format: "%.2f", result.elapsedSeconds)
        let rtf = String(format: "%.4f", result.realTimeFactor)
        let rtfx = String(format: "%.2f", result.durationSeconds / max(result.elapsedSeconds, 1e-6))

        logger.info(String(repeating: "=", count: 50))
        logger.info("GRANITE NAR TRANSCRIPTION")
        logger.info(String(repeating: "=", count: 50))
        print(result.text)
        logger.info("")
        logger.info("Performance:")
        logger.info("  Mode: \(mode.rawValue)")
        logger.info("  Window: \(result.windowSeconds)s + \(overlap)s overlap")
        logger.info("  Chunks: \(result.chunkCount)")
        logger.info("  Audio duration: \(audioDuration)s")
        logger.info("  Processing time: \(elapsed)s")
        logger.info("  RTF: \(rtf)")
        logger.info("  RTFx: \(rtfx)x")

        if let reference {
            printEvaluation(hypothesis: result.text, reference: reference)
        }
    }

    private static func printEvaluation(hypothesis: String, reference: String) {
        let metrics = calculateReferenceMetrics(hypothesis: hypothesis, reference: reference)
        logger.info("")
        logger.info("Evaluation:")
        logger.info("  WER: \(String(format: "%.2f", metrics.wer * 100))%")
        logger.info("  CER: \(String(format: "%.2f", metrics.cer * 100))%")
        let edits = "I \(metrics.insertions), D \(metrics.deletions), S \(metrics.substitutions)"
        logger.info(
            "  Edits: \(edits), words \(metrics.referenceWords)"
        )
    }

    private static func calculateReferenceMetrics(
        hypothesis: String,
        reference: String
    ) -> GraniteReferenceMetrics {
        let hypWords = wordTokens(hypothesis)
        let refWords = wordTokens(reference)
        let wordDistance = editDistance(hypWords, refWords)
        let hypCharacters = characterTokens(hypothesis)
        let refCharacters = characterTokens(reference)
        let characterDistance = editDistance(hypCharacters, refCharacters)

        return GraniteReferenceMetrics(
            wer: refWords.isEmpty ? 0.0 : Double(wordDistance.total) / Double(refWords.count),
            cer: refCharacters.isEmpty ? 0.0 : Double(characterDistance.total) / Double(refCharacters.count),
            insertions: wordDistance.insertions,
            deletions: wordDistance.deletions,
            substitutions: wordDistance.substitutions,
            referenceWords: refWords.count
        )
    }

    private static func wordTokens(_ text: String) -> [String] {
        text.lowercased()
            .split { character in
                !character.isLetter && !character.isNumber
            }
            .map(String.init)
    }

    private static func characterTokens(_ text: String) -> [Character] {
        text.lowercased().filter { character in
            character.isLetter || character.isNumber
        }
    }

    private static func parseMode(_ value: String) -> GraniteAsrMode? {
        switch value.lowercased() {
        case "balanced":
            return .balanced
        case "speed":
            return .speed
        case "long-speed", "longspeed", "long_speed":
            return .longSpeed
        default:
            return nil
        }
    }

    private static func parseComputeUnits(_ value: String) -> MLComputeUnits? {
        switch value.lowercased() {
        case "cpu", "cpu-only", "cpu_only":
            return .cpuOnly
        case "gpu", "cpu-and-gpu", "cpu_and_gpu":
            return .cpuAndGPU
        case "ane", "ne", "cpu-and-neural-engine", "cpu_and_neural_engine":
            return .cpuAndNeuralEngine
        case "all":
            return .all
        default:
            return nil
        }
    }

    private static func printUsage() {
        logger.info(
            """

            Granite NAR Transcribe Command

            Usage: fluidaudio granite-transcribe <audio_file> --model-dir <bundle> [options]

            Options:
                --help, -h                  Show this help message
                --model-dir <path>          Local Granite NAR CoreML bundle
                --reference <txt>           Reference transcript; prints WER/CER
                --mode <balanced|speed|long-speed>
                                            balanced: 35s/5s overlap
                                            speed: 35s for <=5min, then 60s
                                            long-speed: force 60s/5s overlap
                --compute-units <cpu|gpu|ane|all>
                                            Default: gpu

            Example:
                fluidaudio granite-transcribe audio.wav --model-dir /path/to/fluid_audio_granite_nar
            """
        )
    }
}

private struct GraniteReferenceMetrics {
    let wer: Double
    let cer: Double
    let insertions: Int
    let deletions: Int
    let substitutions: Int
    let referenceWords: Int
}

private struct GraniteEditDistance {
    let total: Int
    let insertions: Int
    let deletions: Int
    let substitutions: Int
}

private func editDistance<T: Equatable>(_ hypothesis: [T], _ reference: [T]) -> GraniteEditDistance {
    let hypCount = hypothesis.count
    let refCount = reference.count
    if hypCount == 0 {
        return GraniteEditDistance(total: refCount, insertions: 0, deletions: refCount, substitutions: 0)
    }
    if refCount == 0 {
        return GraniteEditDistance(total: hypCount, insertions: hypCount, deletions: 0, substitutions: 0)
    }

    var table = Array(repeating: Array(repeating: 0, count: refCount + 1), count: hypCount + 1)
    for hypIndex in 0 ... hypCount {
        table[hypIndex][0] = hypIndex
    }
    for refIndex in 0 ... refCount {
        table[0][refIndex] = refIndex
    }

    for hypIndex in 1 ... hypCount {
        for refIndex in 1 ... refCount {
            if hypothesis[hypIndex - 1] == reference[refIndex - 1] {
                table[hypIndex][refIndex] = table[hypIndex - 1][refIndex - 1]
            } else {
                table[hypIndex][refIndex] = 1 + min(
                    table[hypIndex - 1][refIndex],
                    min(table[hypIndex][refIndex - 1], table[hypIndex - 1][refIndex - 1])
                )
            }
        }
    }

    return backtraceEditDistance(table, hypothesis: hypothesis, reference: reference)
}

private func backtraceEditDistance<T: Equatable>(
    _ table: [[Int]],
    hypothesis: [T],
    reference: [T]
) -> GraniteEditDistance {
    var hypIndex = hypothesis.count
    var refIndex = reference.count
    var insertions = 0
    var deletions = 0
    var substitutions = 0

    while hypIndex > 0 || refIndex > 0 {
        if hypIndex > 0, refIndex > 0, hypothesis[hypIndex - 1] == reference[refIndex - 1] {
            hypIndex -= 1
            refIndex -= 1
        } else if hypIndex > 0, refIndex > 0,
                  table[hypIndex][refIndex] == table[hypIndex - 1][refIndex - 1] + 1 {
            substitutions += 1
            hypIndex -= 1
            refIndex -= 1
        } else if hypIndex > 0, table[hypIndex][refIndex] == table[hypIndex - 1][refIndex] + 1 {
            insertions += 1
            hypIndex -= 1
        } else if refIndex > 0, table[hypIndex][refIndex] == table[hypIndex][refIndex - 1] + 1 {
            deletions += 1
            refIndex -= 1
        } else {
            break
        }
    }

    return GraniteEditDistance(
        total: table[hypothesis.count][reference.count],
        insertions: insertions,
        deletions: deletions,
        substitutions: substitutions
    )
}
#endif
