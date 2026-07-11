#if os(macOS)
import FluidAudio
import Foundation

/// Measures the opt-in Parakeet pronunciation customization matcher with real encoder features.
enum PronunciationBenchmarkCommand {
    private static let logger = AppLogger(category: "PronunciationBenchmark")
    private static let prototypeCounts = [1, 10, 50, 100]

    static func run(arguments: [String]) async {
        do {
            let options = try Options(arguments: arguments)
            if options.showHelp {
                printUsage()
                return
            }
            try await run(options: options)
        } catch {
            logger.error("PRONUNCIATION_BENCH_ERROR \(error.localizedDescription)")
            printUsage()
        }
    }

    private static func run(options: Options) async throws {
        let samples = try AudioConverter().resampleAudioFile(path: options.audioPath)
        guard samples.count >= 16_000 else { throw BenchmarkError.audioTooShort }
        guard samples.count <= 240_000 else { throw BenchmarkError.audioTooLong }

        let models = try await AsrModels.downloadAndLoad(version: .v2)
        let manager = AsrManager(
            config: ASRConfig(
                tdtConfig: TdtConfig(blankId: AsrModelVersion.v2.blankId),
                encoderHiddenSize: AsrModelVersion.v2.encoderHiddenSize
            ))
        try await manager.initialize(models: models)
        await manager.setPronunciationCustomizationEnabled(true)

        let transcriptionStarted = ContinuousClock.now
        let result = try await manager.transcribe(samples)
        let transcriptionMicroseconds = microseconds(transcriptionStarted.duration(to: .now))
        guard let features = await manager.consumePronunciationEncoderFeatures() else {
            throw BenchmarkError.encoderFeaturesMissing
        }
        let prototypes = try makePrototypes(
            count: prototypeCounts.last ?? 0,
            sourceFrameCount: options.prototypeFrames,
            useVariedFrameCounts: options.useVariedPrototypeFrames,
            from: features
        )

        let audioMilliseconds = Int((Double(samples.count) / 16).rounded())
        let featureBytes = features.values.count * MemoryLayout<Float>.stride
        let prototypeBytes = prototypes.reduce(0) { $0 + $1.values.count * MemoryLayout<Float>.stride }
        print(
            "PRONUNCIATION_BENCH_INPUT audioMs=\(audioMilliseconds) samples=\(samples.count) "
                + "transcriptChars=\(result.text.count) frames=\(features.frameCount) hidden=\(features.hiddenSize) "
                + "prototypeFrames=\(options.useVariedPrototypeFrames ? "varied4-16" : String(options.prototypeFrames)) "
                + "runs=\(options.runs)"
        )
        print(
            "PRONUNCIATION_BENCH_CAPTURE asrMs=\(formatMilliseconds(transcriptionMicroseconds)) "
                + "captureUs=\(features.captureMicroseconds) featureBytes=\(featureBytes) "
                + "prototypeBytes100=\(prototypeBytes)"
        )

        for count in prototypeCounts {
            let selected = Array(prototypes.prefix(count))
            for _ in 0..<options.warmupRuns {
                _ = PronunciationEmbeddingMatcher.bestMatches(prototypes: selected, in: features)
            }

            var timings: [Int] = []
            timings.reserveCapacity(options.runs)
            for _ in 0..<options.runs {
                let started = ContinuousClock.now
                let matches = PronunciationEmbeddingMatcher.bestMatches(prototypes: selected, in: features)
                timings.append(microseconds(started.duration(to: .now)))
                guard matches.count == count else { throw BenchmarkError.incompleteResults }
            }
            timings.sort()
            print(
                "PRONUNCIATION_BENCH_RESULT prototypes=\(count) medianUs=\(percentile(timings, 0.50)) "
                    + "p95Us=\(percentile(timings, 0.95)) minUs=\(timings.first ?? 0) maxUs=\(timings.last ?? 0)"
            )
        }

        await manager.cleanup()
    }

    private static func makePrototypes(
        count: Int,
        sourceFrameCount: Int,
        useVariedFrameCounts: Bool,
        from features: EncoderFeatureSequence
    ) throws -> [PronunciationEmbedding] {
        return try (0..<count).map { index in
            let requestedFrameCount = useVariedFrameCounts ? 4 + (index % 13) : sourceFrameCount
            let frameCount = min(max(2, requestedFrameCount), features.frameCount)
            let availableStarts = max(1, features.frameCount - frameCount + 1)
            let start = index % availableStarts
            guard
                let embedding = PronunciationEmbeddingMatcher.embedding(
                    from: features,
                    frameRange: start..<(start + frameCount)
                )
            else {
                throw BenchmarkError.prototypeCreationFailed
            }
            return embedding
        }
    }

    private static func percentile(_ sortedValues: [Int], _ percentile: Double) -> Int {
        guard !sortedValues.isEmpty else { return 0 }
        let index = Int((Double(sortedValues.count - 1) * percentile).rounded(.up))
        return sortedValues[min(sortedValues.count - 1, max(0, index))]
    }

    private static func microseconds(_ duration: Duration) -> Int {
        Int(
            (Double(duration.components.seconds) * 1_000_000 + Double(duration.components.attoseconds) / 1e12)
                .rounded()
        )
    }

    private static func formatMilliseconds(_ microseconds: Int) -> String {
        String(format: "%.2f", Double(microseconds) / 1_000)
    }

    private static func printUsage() {
        print(
            """
            Usage:
              fluidaudiocli pronunciation-benchmark <audio-file> [options]

            Options:
              --runs <n>             Measured runs per prototype count (default: 20)
              --warmup-runs <n>      Warm-up runs per prototype count (default: 3)
              --prototype-frames <n> Enrollment duration in 80 ms frames (default: 8)
              --varied-prototype-frames
                                      Cycle through 4–16 frame enrollment durations
              --help                 Show this help

            The command uses real Parakeet encoder features and measures 1, 10, 50, and 100 prototypes.
            Input audio must be between 1 and 15 seconds.
            """
        )
    }
}

extension PronunciationBenchmarkCommand {
    fileprivate struct Options {
        let audioPath: String
        var runs = 20
        var warmupRuns = 3
        var prototypeFrames = 8
        var useVariedPrototypeFrames = false
        var showHelp = false

        init(arguments: [String]) throws {
            if arguments.contains("--help") || arguments.contains("-h") {
                self.audioPath = arguments.first ?? ""
                self.showHelp = true
                return
            }
            guard let audioPath = arguments.first, !audioPath.hasPrefix("--") else {
                throw BenchmarkError.missingAudioPath
            }
            self.audioPath = audioPath

            var index = 1
            while index < arguments.count {
                let option = arguments[index]
                switch option {
                case "--runs":
                    self.runs = try Self.positiveInteger(after: &index, in: arguments, option: option)
                case "--warmup-runs":
                    self.warmupRuns = try Self.nonnegativeInteger(after: &index, in: arguments, option: option)
                case "--prototype-frames":
                    self.prototypeFrames = try Self.positiveInteger(after: &index, in: arguments, option: option)
                case "--varied-prototype-frames":
                    self.useVariedPrototypeFrames = true
                default:
                    throw BenchmarkError.unknownOption(option)
                }
                index += 1
            }
        }

        private static func positiveInteger(
            after index: inout Int, in arguments: [String], option: String
        ) throws
            -> Int
        {
            let parsed = try integer(after: &index, in: arguments, option: option)
            guard parsed > 0 else { throw BenchmarkError.invalidValue(option) }
            return parsed
        }

        private static func nonnegativeInteger(
            after index: inout Int, in arguments: [String], option: String
        ) throws
            -> Int
        {
            let parsed = try integer(after: &index, in: arguments, option: option)
            guard parsed >= 0 else { throw BenchmarkError.invalidValue(option) }
            return parsed
        }

        private static func integer(after index: inout Int, in arguments: [String], option: String) throws -> Int {
            guard index + 1 < arguments.count else { throw BenchmarkError.missingValue(option) }
            index += 1
            guard let parsed = Int(arguments[index]) else { throw BenchmarkError.invalidValue(option) }
            return parsed
        }
    }

    fileprivate enum BenchmarkError: LocalizedError {
        case missingAudioPath
        case missingValue(String)
        case invalidValue(String)
        case unknownOption(String)
        case audioTooShort
        case audioTooLong
        case encoderFeaturesMissing
        case prototypeCreationFailed
        case incompleteResults

        var errorDescription: String? {
            switch self {
            case .missingAudioPath: "An input audio file is required."
            case .missingValue(let option): "\(option) requires a value."
            case .invalidValue(let option): "\(option) requires a valid integer."
            case .unknownOption(let option): "Unknown option: \(option)."
            case .audioTooShort: "Input audio must be at least one second."
            case .audioTooLong: "Input audio must be no longer than 15 seconds."
            case .encoderFeaturesMissing: "Parakeet did not return captured encoder features."
            case .prototypeCreationFailed: "Could not create a pronunciation prototype."
            case .incompleteResults: "The matcher returned an incomplete result set."
            }
        }
    }
}
#endif
