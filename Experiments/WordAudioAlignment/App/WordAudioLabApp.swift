import SwiftUI

@main
struct WordAudioLabApp: App {
    @StateObject private var model = WordAudioLabModel()

    var body: some Scene {
        WindowGroup("Word Audio Lab") {
            WordAudioLabView(model: model)
                .frame(minWidth: 820, minHeight: 760)
        }
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(after: .newItem) {
                Button("Open Audio…") {
                    model.openAudioFile()
                }
                .keyboardShortcut("o", modifiers: .command)
                .disabled(!model.canLoadAudio)
            }

            CommandMenu("Experiment") {
                Button(model.isRecordingSentence ? "Stop Recording" : "Start Recording") {
                    Task { await model.toggleRecording() }
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])
                .disabled(!model.canToggleRecording)

                Button(model.isRecordingEnrollment ? "Stop Enrollment" : "Record Enrollment") {
                    Task { await model.toggleEnrollmentRecording() }
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])
                .disabled(!model.canToggleEnrollment)

                Button("Play Full Recording") {
                    model.playFullRecording()
                }
                .keyboardShortcut("p", modifiers: [.command, .shift])
                .disabled(model.fullSamples.isEmpty)
            }
        }
    }
}
