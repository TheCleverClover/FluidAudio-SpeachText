#if os(macOS)
@preconcurrency import AVFoundation
import FluidAudio
import Foundation

enum WordAudioProbeCommand {
    private static let logger = AppLogger(category: "WordAudioProbe")

    static func run(arguments: [String]) async {
        do {
            let options = try Options(arguments: arguments)
            if options.showHelp {
                printUsage()
                return
            }
            try await run(options: options)
        } catch {
            logger.error("WORD_CHUNK_ERROR \(error.localizedDescription)")
            printUsage()
        }
    }

    private static func run(options: Options) async throws {
        let totalStarted = ContinuousClock.now
        let audioURL = URL(fileURLWithPath: options.audioPath)
        let samples = try AudioConverter().resampleAudioFile(path: options.audioPath)
        let audioDuration = Double(samples.count) / 16_000

        logger.info(
            "WORD_CHUNK_INPUT file=\(audioURL.path) audioMs=\(Int((audioDuration * 1_000).rounded())) "
                + "samples=\(samples.count) target=\(options.word) occurrence=\(options.occurrence)"
        )

        let models = try await AsrModels.downloadAndLoad(version: options.modelVersion)
        let config = ASRConfig(
            tdtConfig: TdtConfig(blankId: options.modelVersion.blankId),
            encoderHiddenSize: options.modelVersion.encoderHiddenSize
        )
        let manager = AsrManager(config: config)
        try await manager.initialize(models: models)

        let transcriptionStarted = ContinuousClock.now
        let result = try await manager.transcribe(samples)
        let transcriptionDuration = transcriptionStarted.duration(to: .now)
        let tokenTimings = result.tokenTimings ?? []
        let words = WordAudioChunkExtractor.words(from: tokenTimings)

        logger.info("WORD_CHUNK_TRANSCRIPT text=\(result.text)")
        for (index, word) in words.enumerated() {
            logger.info(
                "WORD_CHUNK_WORD index=\(index) text=\(word.text) startMs=\(milliseconds(word.startTime)) "
                    + "endMs=\(milliseconds(word.endTime)) confidence=\(format(word.confidence))"
            )
        }

        let extractionStarted = ContinuousClock.now
        let chunk = try WordAudioChunkExtractor.extract(
            word: options.word,
            occurrence: options.occurrence,
            tokenTimings: tokenTimings,
            audioSamples: samples,
            padding: Double(options.paddingMilliseconds) / 1_000
        )
        let extractionDuration = extractionStarted.duration(to: .now)

        let outputURL = options.outputURL(audioURL: audioURL)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let wavData = try AudioWAV.data(from: chunk.samples, sampleRate: 16_000)
        try wavData.write(to: outputURL, options: .atomic)

        logger.info(
            "WORD_CHUNK_SELECTED word=\(chunk.word.text) wordIndex=\(chunk.wordIndex) "
                + "rawStartMs=\(milliseconds(chunk.word.startTime)) rawEndMs=\(milliseconds(chunk.word.endTime)) "
                + "extractStartMs=\(milliseconds(chunk.extractionStartTime)) "
                + "extractEndMs=\(milliseconds(chunk.extractionEndTime)) "
                + "sampleStart=\(chunk.sampleRange.lowerBound) sampleEnd=\(chunk.sampleRange.upperBound)"
        )
        logger.info(
            "WORD_CHUNK_TIMING transcriptionMs=\(milliseconds(transcriptionDuration)) "
                + "extractionUs=\(microseconds(extractionDuration)) totalMs=\(milliseconds(totalStarted.duration(to: .now)))"
        )
        logger.info("WORD_CHUNK_OUTPUT path=\(outputURL.path) bytes=\(wavData.count)")

        if options.play {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
            process.arguments = [outputURL.path]
            try process.run()
            process.waitUntilExit()
        }

        await manager.cleanup()
    }

    private static func milliseconds(_ seconds: TimeInterval) -> Int {
        Int((seconds * 1_000).rounded())
    }

    private static func milliseconds(_ duration: Duration) -> Int {
        Int((Double(duration.components.seconds) * 1_000 + Double(duration.components.attoseconds) / 1e15).rounded())
    }

    private static func microseconds(_ duration: Duration) -> Int {
        Int(
            (Double(duration.components.seconds) * 1_000_000 + Double(duration.components.attoseconds) / 1e12).rounded()
        )
    }

    private static func format(_ value: Float) -> String {
        String(format: "%.3f", value)
    }

    private static func printUsage() {
        print(
            """
            Usage:
              fluidaudiocli word-audio-probe <audio-file> --word <word> [options]

            Options:
              --word <word>            Transcribed word to extract (required)
              --occurrence <n>         Select the nth matching word (default: 1)
              --padding-ms <ms>        Add listening context on both sides (default: 120)
              --output <wav-path>      Output path (default: <input>-<word>-chunk.wav)
              --model-version <v2|v3>  Parakeet model (default: v2)
              --play                   Play the extracted WAV after writing it
              --help                   Show this help
            """
        )
    }
}

extension WordAudioProbeCommand {
    fileprivate struct Options {
        let audioPath: String
        var word = ""
        var occurrence = 1
        var paddingMilliseconds = 120
        var outputPath: String?
        var modelVersion: AsrModelVersion = .v2
        var play = false
        var showHelp = false

        init(arguments: [String]) throws {
            if arguments.contains("--help") || arguments.contains("-h") {
                self.audioPath = arguments.first ?? ""
                self.showHelp = true
                return
            }
            guard let audioPath = arguments.first, !audioPath.hasPrefix("--") else {
                throw ProbeArgumentError.missingAudioPath
            }
            self.audioPath = audioPath

            var index = 1
            while index < arguments.count {
                switch arguments[index] {
                case "--word":
                    self.word = try Self.value(after: &index, in: arguments, option: "--word")
                case "--occurrence":
                    let value = try Self.value(after: &index, in: arguments, option: "--occurrence")
                    guard let parsed = Int(value), parsed > 0 else { throw ProbeArgumentError.invalidOccurrence(value) }
                    self.occurrence = parsed
                case "--padding-ms":
                    let value = try Self.value(after: &index, in: arguments, option: "--padding-ms")
                    guard let parsed = Int(value), parsed >= 0 else { throw ProbeArgumentError.invalidPadding(value) }
                    self.paddingMilliseconds = parsed
                case "--output":
                    self.outputPath = try Self.value(after: &index, in: arguments, option: "--output")
                case "--model-version":
                    let value = try Self.value(after: &index, in: arguments, option: "--model-version")
                    switch value.lowercased() {
                    case "v2", "2": self.modelVersion = .v2
                    case "v3", "3": self.modelVersion = .v3
                    default: throw ProbeArgumentError.invalidModelVersion(value)
                    }
                case "--play":
                    self.play = true
                default:
                    throw ProbeArgumentError.unknownOption(arguments[index])
                }
                index += 1
            }

            guard !self.word.isEmpty else { throw ProbeArgumentError.missingWord }
        }

        func outputURL(audioURL: URL) -> URL {
            if let outputPath {
                return URL(fileURLWithPath: outputPath)
            }
            let safeWord = word.map { $0.isLetter || $0.isNumber ? String($0) : "-" }.joined()
            return audioURL.deletingPathExtension().appendingPathExtension("\(safeWord)-chunk.wav")
        }

        private static func value(after index: inout Int, in arguments: [String], option: String) throws -> String {
            guard index + 1 < arguments.count else { throw ProbeArgumentError.missingValue(option) }
            index += 1
            return arguments[index]
        }
    }

    fileprivate enum ProbeArgumentError: LocalizedError {
        case missingAudioPath
        case missingWord
        case missingValue(String)
        case invalidOccurrence(String)
        case invalidPadding(String)
        case invalidModelVersion(String)
        case unknownOption(String)

        var errorDescription: String? {
            switch self {
            case .missingAudioPath: "An input audio file is required."
            case .missingWord: "--word is required."
            case .missingValue(let option): "\(option) requires a value."
            case .invalidOccurrence(let value): "Invalid occurrence: \(value)."
            case .invalidPadding(let value): "Invalid padding: \(value)."
            case .invalidModelVersion(let value): "Invalid model version: \(value). Use v2 or v3."
            case .unknownOption(let option): "Unknown option: \(option)."
            }
        }
    }
}
#endif
