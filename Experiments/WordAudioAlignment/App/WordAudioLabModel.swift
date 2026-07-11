@preconcurrency import AVFoundation
import AppKit
import Combine
import FluidAudio
import Foundation
import UniformTypeIdentifiers

@MainActor
final class WordAudioLabModel: ObservableObject {
    private let logger = AppLogger(category: "WordAudioLab")

    enum RecordingPurpose {
        case sentence
        case enrollment
    }

    enum State: Equatable {
        case loadingModel
        case ready
        case recording
        case transcribing
        case failed(String)

        var label: String {
            switch self {
            case .loadingModel: "Loading Parakeet v2…"
            case .ready: "Ready"
            case .recording: "Recording…"
            case .transcribing: "Splitting words…"
            case .failed(let message): message
            }
        }
    }

    struct Segment: Identifiable {
        let id: Int
        let word: String
        let startTime: TimeInterval
        let endTime: TimeInterval
        let confidence: Float
        let samples: [Float]
    }

    struct EnrollmentSample: Identifiable {
        let id = UUID()
        let label: String
        let decodedText: String
        let frameCount: Int
        let samples: [Float]
        let embedding: PronunciationEmbedding
    }

    struct MatchProposal: Identifiable {
        let id = UUID()
        let label: String
        let score: Float
        let startTime: TimeInterval
        let endTime: TimeInterval
        let originalWords: String
        let replacementPreview: String
        let samples: [Float]
        let accepted: Bool
    }

    @Published private(set) var state: State = .loadingModel
    @Published private(set) var transcript = ""
    @Published private(set) var fullSamples: [Float] = []
    @Published private(set) var segments: [Segment] = []
    @Published private(set) var recordingDuration: TimeInterval = 0
    @Published private(set) var transcriptionMilliseconds = 0
    @Published private(set) var segmentationMicroseconds = 0
    @Published private(set) var matchingMicroseconds = 0
    @Published private(set) var capturedFeatureBytes = 0
    @Published private(set) var enrollmentSamples: [EnrollmentSample] = []
    @Published private(set) var matchProposals: [MatchProposal] = []
    @Published private(set) var correctedPreview = ""
    @Published var enrollmentLabel = ""
    @Published var matchThreshold = Double(PronunciationCustomizationDefaults.acceptanceThreshold) {
        didSet { rebuildMatches() }
    }
    @Published var listeningPaddingMilliseconds = 80 {
        didSet { rebuildSegments() }
    }

    private var manager: AsrManager?
    private var recorder: AVAudioRecorder?
    private var player: AVAudioPlayer?
    private var tokenTimings: [TokenTiming] = []
    private var sentenceFeatures: EncoderFeatureSequence?
    private var recordingPurpose: RecordingPurpose?
    private var activeEnrollmentLabel = ""
    private var recordingStartedAt: ContinuousClock.Instant?
    private var durationTask: Task<Void, Never>?

    var isRecording: Bool { state == .recording }
    var isRecordingSentence: Bool { isRecording && recordingPurpose == .sentence }
    var isRecordingEnrollment: Bool { isRecording && recordingPurpose == .enrollment }
    private var isReadyForInput: Bool {
        guard manager != nil else { return false }
        return state == .ready || isRecoverableFailure
    }

    private var isRecoverableFailure: Bool {
        if case .failed = state { return true }
        return false
    }

    var canToggleRecording: Bool { isReadyForInput || isRecordingSentence }
    var canToggleEnrollment: Bool {
        isRecordingEnrollment
            || (isReadyForInput && !enrollmentLabel.trimmingCharacters(in: .whitespaces).isEmpty)
    }
    var canLoadAudio: Bool { isReadyForInput }

    init() {
        Task { await loadModel() }
    }

    func toggleRecording() async {
        if isRecordingSentence {
            await stopAndTranscribe()
        } else {
            await startRecording(purpose: .sentence)
        }
    }

    func toggleEnrollmentRecording() async {
        if isRecordingEnrollment {
            await stopAndTranscribe()
        } else {
            await startRecording(purpose: .enrollment)
        }
    }

    func clearEnrollments() {
        enrollmentSamples = []
        matchProposals = []
        correctedPreview = transcript
    }

    func play(enrollment: EnrollmentSample) {
        play(samples: enrollment.samples)
    }

    func play(proposal: MatchProposal) {
        play(samples: proposal.samples)
    }

    func playFullRecording() {
        play(samples: fullSamples)
    }

    func openAudioFile() {
        guard canLoadAudio else { return }
        let panel = NSOpenPanel()
        panel.title = "Choose a speech recording"
        panel.prompt = "Analyze"
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.audio]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await processAudio(at: url) }
    }

    func openEnrollmentAudioFile() {
        let label = enrollmentLabel.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isReadyForInput, !label.isEmpty else { return }
        let panel = NSOpenPanel()
        panel.title = "Choose an isolated pronunciation"
        panel.prompt = "Enroll"
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.audio]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await processEnrollmentAudio(at: url, label: label) }
    }

    func play(segment: Segment) {
        play(samples: segment.samples)
    }

    func stopPlayback() {
        player?.stop()
        player = nil
    }

    private func loadModel() async {
        do {
            let models = try await AsrModels.downloadAndLoad(version: .v2)
            let config = ASRConfig(
                tdtConfig: TdtConfig(blankId: AsrModelVersion.v2.blankId),
                encoderHiddenSize: AsrModelVersion.v2.encoderHiddenSize
            )
            let manager = AsrManager(config: config)
            try await manager.initialize(models: models)
            await manager.setPronunciationCustomizationEnabled(true)
            self.manager = manager
            self.state = .ready
        } catch {
            self.state = .failed("Model error: \(error.localizedDescription)")
        }
    }

    private func startRecording(purpose: RecordingPurpose) async {
        guard await microphoneAccessGranted() else {
            state = .failed("Microphone access is required")
            return
        }

        do {
            stopPlayback()
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("word-audio-lab-recording.wav")
            try? FileManager.default.removeItem(at: url)
            let settings: [String: Any] = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: 48_000,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsBigEndianKey: false,
            ]
            let recorder = try AVAudioRecorder(url: url, settings: settings)
            recorder.prepareToRecord()
            guard recorder.record() else {
                throw LabError.recordingDidNotStart
            }
            self.recorder = recorder
            if purpose == .sentence {
                transcript = ""
                fullSamples = []
                tokenTimings = []
                segments = []
                sentenceFeatures = nil
                matchProposals = []
                correctedPreview = ""
            } else {
                activeEnrollmentLabel = enrollmentLabel.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            transcriptionMilliseconds = 0
            segmentationMicroseconds = 0
            recordingDuration = 0
            recordingStartedAt = .now
            recordingPurpose = purpose
            state = .recording
            startDurationUpdates()
        } catch {
            state = .failed("Recording error: \(error.localizedDescription)")
        }
    }

    private func stopAndTranscribe() async {
        guard let recorder else { return }
        recorder.stop()
        self.recorder = nil
        durationTask?.cancel()
        durationTask = nil
        recordingStartedAt = nil
        let purpose = recordingPurpose
        recordingPurpose = nil
        switch purpose {
        case .enrollment:
            await processEnrollmentAudio(at: recorder.url, label: activeEnrollmentLabel)
        case .sentence, .none:
            await processAudio(at: recorder.url)
        }
    }

    private func processAudio(at url: URL) async {
        state = .transcribing
        do {
            let samples = try AudioConverter().resampleAudioFile(url)
            let (result, features) = try await transcribeWithFeatures(samples: samples)
            fullSamples = samples
            transcript = result.text
            tokenTimings = result.tokenTimings ?? []
            sentenceFeatures = features
            capturedFeatureBytes = features.values.count * MemoryLayout<Float>.size
            rebuildSegments()
            rebuildMatches()
            logger.info(
                "LAB_SENTENCE audioMs=\(Int((Double(samples.count) / 16).rounded())) "
                    + "asrMs=\(transcriptionMilliseconds) frames=\(features.frameCount) hidden=\(features.hiddenSize) "
                    + "captureUs=\(features.captureMicroseconds) bytes=\(capturedFeatureBytes) words=\(segments.count)"
            )
            state = .ready
        } catch {
            state = .failed("Transcription error: \(error.localizedDescription)")
        }
    }

    private func processEnrollmentAudio(at url: URL, label: String) async {
        state = .transcribing
        do {
            guard !label.isEmpty else { throw LabError.missingEnrollmentLabel }
            let samples = try AudioConverter().resampleAudioFile(url)
            let (result, features) = try await transcribeWithFeatures(samples: samples)
            let timings = result.tokenTimings ?? []
            let frameRange = enrollmentFrameRange(tokenTimings: timings, features: features)
            guard let embedding = PronunciationEmbeddingMatcher.embedding(from: features, frameRange: frameRange) else {
                throw LabError.embeddingFailed
            }
            let speechSamples = sliceSamples(
                samples,
                startTime: Double(frameRange.lowerBound) * features.frameDuration,
                endTime: Double(frameRange.upperBound) * features.frameDuration
            )
            enrollmentSamples.append(
                EnrollmentSample(
                    label: label,
                    decodedText: result.text,
                    frameCount: embedding.sourceFrameCount,
                    samples: speechSamples,
                    embedding: embedding
                ))
            logger.info(
                "LAB_ENROLL label=\(label) decoded=\(result.text) frames=\(embedding.sourceFrameCount) "
                    + "asrMs=\(transcriptionMilliseconds) captureUs=\(features.captureMicroseconds) "
                    + "enrollments=\(enrollmentSamples.filter { $0.label == label }.count)"
            )
            rebuildMatches()
            state = .ready
        } catch {
            state = .failed("Enrollment error: \(error.localizedDescription)")
        }
    }

    private func transcribeWithFeatures(samples: [Float]) async throws -> (ASRResult, EncoderFeatureSequence) {
        guard let manager else { throw LabError.modelNotReady }
        guard samples.count >= 16_000 else { throw LabError.recordingTooShort }
        guard samples.count <= 240_000 else { throw LabError.recordingTooLong }

        let transcriptionStart = ContinuousClock.now
        let result = try await manager.transcribe(samples)
        transcriptionMilliseconds = Self.milliseconds(transcriptionStart.duration(to: .now))
        guard let features = await manager.consumePronunciationEncoderFeatures() else {
            throw LabError.encoderFeaturesMissing
        }
        return (result, features)
    }

    private func enrollmentFrameRange(
        tokenTimings: [TokenTiming],
        features: EncoderFeatureSequence
    ) -> Range<Int> {
        guard let first = tokenTimings.first, let last = tokenTimings.last else {
            return 0..<features.frameCount
        }
        let start = max(0, min(features.frameCount - 1, Int(floor(first.startTime / features.frameDuration))))
        let end = max(start + 1, min(features.frameCount, Int(ceil(last.endTime / features.frameDuration))))
        return start..<end
    }

    private func rebuildSegments() {
        guard !fullSamples.isEmpty, !tokenTimings.isEmpty else {
            segments = []
            return
        }
        let startedAt = ContinuousClock.now
        let words = WordAudioChunkExtractor.words(from: tokenTimings)
        segments = words.indices.compactMap { index in
            guard
                let chunk = try? WordAudioChunkExtractor.extract(
                    wordAt: index,
                    tokenTimings: tokenTimings,
                    audioSamples: fullSamples,
                    padding: Double(listeningPaddingMilliseconds) / 1_000
                )
            else { return nil }
            return Segment(
                id: index,
                word: chunk.word.text,
                startTime: chunk.word.startTime,
                endTime: chunk.word.endTime,
                confidence: chunk.word.confidence,
                samples: chunk.samples
            )
        }
        segmentationMicroseconds = Self.microseconds(startedAt.duration(to: .now))
    }

    private func rebuildMatches() {
        guard let sentenceFeatures, !enrollmentSamples.isEmpty, !tokenTimings.isEmpty else {
            matchProposals = []
            correctedPreview = transcript
            return
        }

        let startedAt = ContinuousClock.now
        let words = WordAudioChunkExtractor.words(from: tokenTimings)
        let grouped = Dictionary(grouping: enrollmentSamples, by: \.label)
        let prototypeEntries: [(label: String, prototype: PronunciationEmbedding)] = grouped.compactMap {
            label, samples in
            guard let prototype = PronunciationEmbeddingMatcher.prototype(from: samples.map(\.embedding)) else {
                return nil
            }
            return (label, prototype)
        }.sorted { $0.label < $1.label }
        let matches = PronunciationEmbeddingMatcher.bestMatches(
            prototypes: prototypeEntries.map(\.prototype),
            in: sentenceFeatures
        )
        var proposals: [MatchProposal] = []

        for (entry, possibleMatch) in zip(prototypeEntries, matches) {
            guard let match = possibleMatch else { continue }
            let label = entry.label

            let startTime = Double(match.frameRange.lowerBound) * sentenceFeatures.frameDuration
            let endTime = Double(match.frameRange.upperBound) * sentenceFeatures.frameDuration
            var overlappingIndices = WordAudioChunkExtractor.substantiallyOverlappingWordIndices(
                in: words,
                startTime: startTime,
                endTime: endTime
            )
            if overlappingIndices.isEmpty, !words.isEmpty {
                let midpoint = (startTime + endTime) / 2
                if let nearest = words.indices.min(by: {
                    abs(((words[$0].startTime + words[$0].endTime) / 2) - midpoint)
                        < abs(((words[$1].startTime + words[$1].endTime) / 2) - midpoint)
                }) {
                    overlappingIndices = [nearest]
                }
            }

            let originalWords = overlappingIndices.map { words[$0].text }.joined(separator: " ")
            let preview = replacementPreview(words: words, replacing: overlappingIndices, with: label)
            let accepted = Double(match.score) >= matchThreshold
            proposals.append(
                MatchProposal(
                    label: label,
                    score: match.score,
                    startTime: startTime,
                    endTime: endTime,
                    originalWords: originalWords,
                    replacementPreview: preview,
                    samples: sliceSamples(fullSamples, startTime: startTime, endTime: endTime),
                    accepted: accepted
                ))
            logger.info(
                "LAB_MATCH label=\(label) score=\(String(format: "%.4f", match.score)) "
                    + "threshold=\(String(format: "%.2f", matchThreshold)) startMs=\(Int((startTime * 1_000).rounded())) "
                    + "endMs=\(Int((endTime * 1_000).rounded())) overlap=\(originalWords) accepted=\(accepted)"
            )
        }

        matchProposals = proposals.sorted { $0.score > $1.score }
        correctedPreview = matchProposals.first(where: \.accepted)?.replacementPreview ?? transcript
        matchingMicroseconds = Self.microseconds(startedAt.duration(to: .now))
    }

    private func replacementPreview(
        words: [TimedTranscriptWord],
        replacing indices: [Int],
        with replacement: String
    ) -> String {
        guard let first = indices.first else { return transcript }
        let replacedIndices = Set(indices)
        var output: [String] = []
        for index in words.indices {
            if index == first {
                output.append(replacement)
            } else if !replacedIndices.contains(index) {
                output.append(words[index].text)
            }
        }
        return output.joined(separator: " ")
    }

    private func sliceSamples(
        _ samples: [Float],
        startTime: TimeInterval,
        endTime: TimeInterval
    ) -> [Float] {
        let start = max(0, min(samples.count, Int((startTime * 16_000).rounded())))
        let end = max(start, min(samples.count, Int((endTime * 16_000).rounded())))
        return Array(samples[start..<end])
    }

    private func play(samples: [Float]) {
        guard !samples.isEmpty else { return }
        do {
            stopPlayback()
            let data = try AudioWAV.data(from: samples, sampleRate: 16_000)
            let player = try AVAudioPlayer(data: data)
            player.prepareToPlay()
            player.play()
            self.player = player
        } catch {
            state = .failed("Playback error: \(error.localizedDescription)")
        }
    }

    private func startDurationUpdates() {
        durationTask?.cancel()
        durationTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(100))
                guard let self, let startedAt = self.recordingStartedAt else { return }
                self.recordingDuration = Self.seconds(startedAt.duration(to: .now))
            }
        }
    }

    private func microphoneAccessGranted() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            true
        case .notDetermined:
            await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    continuation.resume(returning: granted)
                }
            }
        default:
            false
        }
    }

    private static func seconds(_ duration: Duration) -> TimeInterval {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    private static func milliseconds(_ duration: Duration) -> Int {
        Int((seconds(duration) * 1_000).rounded())
    }

    private static func microseconds(_ duration: Duration) -> Int {
        Int((seconds(duration) * 1_000_000).rounded())
    }
}

private enum LabError: LocalizedError {
    case modelNotReady
    case recordingDidNotStart
    case recordingTooShort
    case recordingTooLong
    case missingEnrollmentLabel
    case encoderFeaturesMissing
    case embeddingFailed

    var errorDescription: String? {
        switch self {
        case .modelNotReady: "Parakeet is not ready."
        case .recordingDidNotStart: "The audio recorder did not start."
        case .recordingTooShort: "Record at least one second of speech."
        case .recordingTooLong: "Keep experiment recordings under 15 seconds."
        case .missingEnrollmentLabel: "Enter the intended word before recording an enrollment."
        case .encoderFeaturesMissing: "Parakeet did not return captured encoder features."
        case .embeddingFailed: "The captured encoder frames could not produce an embedding."
        }
    }
}
