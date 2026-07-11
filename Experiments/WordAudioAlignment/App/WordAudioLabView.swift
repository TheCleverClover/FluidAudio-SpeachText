import SwiftUI

struct WordAudioLabView: View {
    @ObservedObject var model: WordAudioLabModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            header
            recordingControls
            transcriptSection
            wordSegmentsSection
            embeddingExperimentSection
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "waveform.badge.magnifyingglass")
                .font(.system(size: 28))
                .foregroundStyle(.tint)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text("Word Audio Lab")
                    .font(.title2.weight(.semibold))
                Text("Verify which audio span Parakeet assigned to each transcribed word.")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Label(model.state.label, systemImage: statusSymbol)
                .font(.callout.weight(.medium))
                .foregroundStyle(statusColor)
        }
    }

    private var recordingControls: some View {
        GroupBox("Recording") {
            HStack(spacing: 14) {
                Button {
                    Task { await model.toggleRecording() }
                } label: {
                    Label(
                        model.isRecordingSentence ? "Stop & Split" : "Record",
                        systemImage: model.isRecordingSentence ? "stop.fill" : "mic.fill"
                    )
                    .frame(minWidth: 112)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .tint(model.isRecordingSentence ? .red : .accentColor)
                .disabled(!model.canToggleRecording)
                .accessibilityHint(
                    model.isRecordingSentence
                        ? "Stops recording and splits the transcript into words" : "Starts microphone recording")

                Button {
                    model.openAudioFile()
                } label: {
                    Label("Open Audio…", systemImage: "waveform.badge.plus")
                }
                .controlSize(.large)
                .disabled(!model.canLoadAudio)
                .help("Analyze an existing WAV, M4A, AIFF, or other audio file")

                Text(formatTime(model.recordingDuration))
                    .font(.system(.title3, design: .monospaced).weight(.medium))
                    .foregroundStyle(model.isRecording ? .red : .secondary)

                if model.state == .loadingModel || model.state == .transcribing {
                    ProgressView()
                        .controlSize(.small)
                }
                Spacer()
                if model.transcriptionMilliseconds > 0 {
                    Text("ASR \(model.transcriptionMilliseconds) ms · split \(model.segmentationMicroseconds) µs")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.vertical, 6)
        }
    }

    private var transcriptSection: some View {
        GroupBox("Full sentence") {
            VStack(alignment: .leading, spacing: 12) {
                Text(model.transcript.isEmpty ? "Record a sentence to see its transcript and audio." : model.transcript)
                    .font(.title3)
                    .foregroundStyle(model.transcript.isEmpty ? .secondary : .primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)

                if !model.fullSamples.isEmpty {
                    HStack(spacing: 12) {
                        Button {
                            model.playFullRecording()
                        } label: {
                            Label("Play sentence", systemImage: "play.fill")
                        }
                        .buttonStyle(.bordered)

                        AudioWaveform(samples: model.fullSamples)
                            .frame(height: 52)
                            .accessibilityLabel("Waveform for the full recording")
                    }
                }
            }
            .padding(.vertical, 6)
        }
    }

    private var wordSegmentsSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Word segments")
                            .font(.headline)
                        Text("Play each word to judge whether the model boundary is correct.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Picker("Listening padding", selection: $model.listeningPaddingMilliseconds) {
                        Text("Raw").tag(0)
                        Text("+80 ms").tag(80)
                        Text("+120 ms").tag(120)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 230)
                    .help("Adds context around each raw word boundary for listening only")
                }

                if model.segments.isEmpty {
                    ContentUnavailableView(
                        "No word segments yet",
                        systemImage: "waveform",
                        description: Text("Record and stop to generate one playable segment per word.")
                    )
                    .frame(maxWidth: .infinity, minHeight: 150)
                } else {
                    ScrollView {
                        LazyVStack(spacing: 8) {
                            ForEach(model.segments) { segment in
                                WordSegmentRow(segment: segment) {
                                    model.play(segment: segment)
                                }
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }
            }
            .padding(.vertical, 6)
        }
    }

    private var embeddingExperimentSection: some View {
        GroupBox("Embedding experiment") {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 10) {
                    TextField("Intended spelling, e.g. Bharatwaj", text: $model.enrollmentLabel)
                        .textFieldStyle(.roundedBorder)
                        .disabled(model.isRecording)

                    Button {
                        Task { await model.toggleEnrollmentRecording() }
                    } label: {
                        Label(
                            model.isRecordingEnrollment ? "Stop enrollment" : "Record enrollment",
                            systemImage: model.isRecordingEnrollment ? "stop.fill" : "waveform.badge.plus"
                        )
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(model.isRecordingEnrollment ? .red : .accentColor)
                    .disabled(!model.canToggleEnrollment)

                    Button {
                        model.openEnrollmentAudioFile()
                    } label: {
                        Label("Use Audio…", systemImage: "folder")
                    }
                    .disabled(!model.canToggleEnrollment || model.isRecordingEnrollment)

                    Button("Clear") {
                        model.clearEnrollments()
                    }
                    .disabled(model.enrollmentSamples.isEmpty || model.isRecording)
                }

                if model.enrollmentSamples.isEmpty {
                    Text(
                        "Type the intended word and record it 3–10 times. The decoded spelling is not used for matching."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                } else {
                    ScrollView(.horizontal) {
                        HStack(spacing: 8) {
                            ForEach(Array(model.enrollmentSamples.enumerated()), id: \.element.id) { index, sample in
                                Button {
                                    model.play(enrollment: sample)
                                } label: {
                                    Label(
                                        "\(sample.label) \(index + 1) · \(sample.frameCount)f · heard “\(sample.decodedText)”",
                                        systemImage: "play.fill"
                                    )
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                    }
                }

                Divider()

                HStack {
                    Text("Acceptance threshold")
                        .font(.callout.weight(.medium))
                    Slider(value: $model.matchThreshold, in: 0.5...0.99, step: 0.01)
                    Text(String(format: "%.2f", model.matchThreshold))
                        .font(.callout.monospacedDigit())
                        .frame(width: 38, alignment: .trailing)
                    Spacer()
                    if model.capturedFeatureBytes > 0 {
                        Text(
                            "features \(ByteCountFormatter.string(fromByteCount: Int64(model.capturedFeatureBytes), countStyle: .memory)) · match \(model.matchingMicroseconds) µs"
                        )
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                    }
                }

                if model.matchProposals.isEmpty {
                    Text("After enrollment, record or open a sentence to scan its audio without string matching.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(model.matchProposals) { proposal in
                        HStack(spacing: 10) {
                            Image(systemName: proposal.accepted ? "checkmark.circle.fill" : "xmark.circle")
                                .foregroundStyle(proposal.accepted ? .green : .secondary)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("“\(proposal.originalWords)” → “\(proposal.label)”")
                                    .font(.body.weight(.medium))
                                Text(
                                    "score \(String(format: "%.3f", proposal.score)) · \(milliseconds(proposal.startTime))–\(milliseconds(proposal.endTime)) ms"
                                )
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button {
                                model.play(proposal: proposal)
                            } label: {
                                Label("Play match", systemImage: "play.fill")
                            }
                            .buttonStyle(.bordered)
                        }
                    }

                    VStack(alignment: .leading, spacing: 3) {
                        Text("Replacement preview")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Text(model.correctedPreview)
                            .font(.body)
                            .textSelection(.enabled)
                    }
                }
            }
            .padding(.vertical, 6)
        }
    }

    private var statusSymbol: String {
        switch model.state {
        case .loadingModel, .transcribing: "clock"
        case .ready: "checkmark.circle.fill"
        case .recording: "record.circle.fill"
        case .failed: "exclamationmark.triangle.fill"
        }
    }

    private var statusColor: Color {
        switch model.state {
        case .ready: .green
        case .recording, .failed: .red
        default: .secondary
        }
    }

    private func formatTime(_ duration: TimeInterval) -> String {
        String(format: "%02d:%02d.%01d", Int(duration) / 60, Int(duration) % 60, Int(duration * 10) % 10)
    }

    private func milliseconds(_ seconds: TimeInterval) -> Int {
        Int((seconds * 1_000).rounded())
    }
}

private struct WordSegmentRow: View {
    let segment: WordAudioLabModel.Segment
    let play: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Text("\(segment.id + 1)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 24, alignment: .trailing)

            VStack(alignment: .leading, spacing: 2) {
                Text(segment.word)
                    .font(.body.weight(.semibold))
                    .lineLimit(1)
                Text(
                    "\(milliseconds(segment.startTime))–\(milliseconds(segment.endTime)) ms · \(Int((segment.confidence * 100).rounded()))%"
                )
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            }
            .frame(width: 180, alignment: .leading)

            AudioWaveform(samples: segment.samples)
                .frame(height: 34)
                .accessibilityLabel("Waveform for \(segment.word)")

            Button(action: play) {
                Image(systemName: "play.fill")
            }
            .buttonStyle(.bordered)
            .help("Play \(segment.word)")
            .accessibilityLabel("Play word \(segment.word)")
        }
        .padding(.vertical, 5)
    }

    private func milliseconds(_ seconds: TimeInterval) -> Int {
        Int((seconds * 1_000).rounded())
    }
}
