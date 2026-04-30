#if os(macOS)
@preconcurrency import CoreML
import FluidAudio
import Foundation

private struct GraniteTranscribeOptions {
    let audioFile: String
    var modelDir: String?
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
            printResult(result, mode: options.mode)
        } catch {
            logger.error("Granite transcription failed: \(error)")
        }
    }

    private static func printResult(_ result: GraniteTranscriptionResult, mode: GraniteAsrMode) {
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
#endif
