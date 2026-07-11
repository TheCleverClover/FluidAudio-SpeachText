# On-Device Pronunciation Customization

FluidAudio includes an opt-in pronunciation customization path for adapting Parakeet TDT to names and uncommon
terms without retraining or recompiling the Core ML model. Applications enroll several recordings, store a compact
prototype embedding, and compare that prototype with encoder windows from later transcriptions.

This feature is experimental. It exposes the reusable FluidAudio primitives and a removable lab, but it does not
automatically rewrite transcripts or persist a user's dictionary.

## Architecture

```text
16 kHz audio
    |
    v
Parakeet encoder -----------------------> TDT decoder -> transcript + token timings
    |
    | copied only when explicitly enabled
    v
encoder frames -> pooled embedding -> batched cosine matching -> acoustic time range
                                                            |
                                                            v
                                             substantially overlapping words
```

The matcher reuses the frozen Parakeet encoder output from the normal transcription pass. It does not load a second
model or run another Core ML inference path.

### Enrollment

1. The application asks the user for the intended spelling.
2. The user records an isolated pronunciation 3–10 times.
3. Each recording runs through normal Parakeet transcription with pronunciation capture enabled.
4. Token timings crop leading and trailing silence from the captured encoder frames.
5. `PronunciationEmbeddingMatcher.embedding` mean-pools and L2-normalizes the selected frames.
6. `PronunciationEmbeddingMatcher.prototype` averages the enrollment embeddings into one persistable prototype.

The intended spelling is metadata. It is not compared with Parakeet's decoded spelling during acoustic matching.

### Matching

For each transcription, FluidAudio scans encoder windows near each prototype's enrolled duration. Window sizes are
the enrolled duration and approximately ±25%. All prototypes sharing a window duration are compared in one
Accelerate matrix operation.

The winning encoder range is converted to seconds at 80 ms per frame. Word ownership then requires more than 50% of
each decoded word's duration to overlap the acoustic range. This prevents a small boundary spill from claiming a
previous word while still allowing one enrollment to replace split output such as `Barat watch`.

## Opt-In Lifecycle

Pronunciation capture is disabled on every new `AsrManager`.

```swift
let manager = AsrManager(config: config)
try await manager.initialize(models: models)

// Enable only when the application has enrolled pronunciation prototypes.
await manager.setPronunciationCustomizationEnabled(true)

let result = try await manager.transcribe(samples)
guard let features = await manager.consumePronunciationEncoderFeatures() else {
    return
}

// Disable to restore the normal ASR path.
await manager.setPronunciationCustomizationEnabled(false)
```

When disabled:

- encoder frames are not copied;
- no pronunciation matching runs automatically;
- no prototypes are retained by `AsrManager`;
- normal TDT decoding and transcript output are unchanged.

`consumePronunciationEncoderFeatures()` returns and clears the latest captured sequence. Consumers should call it
after each transcription while the feature is enabled.

## Building and Storing Prototypes

For an isolated enrollment, use its first and last emitted token times to choose the encoder frame range:

```swift
let timings = result.tokenTimings ?? []
guard let first = timings.first, let last = timings.last else { return }

let start = max(0, Int(floor(first.startTime / features.frameDuration)))
let end = min(features.frameCount, Int(ceil(last.endTime / features.frameDuration)))
guard start < end else { return }

guard let enrollment = PronunciationEmbeddingMatcher.embedding(
    from: features,
    frameRange: start..<end
) else { return }
```

Average several recordings for the same label:

```swift
guard let prototype = PronunciationEmbeddingMatcher.prototype(from: enrollments) else {
    return
}
```

`PronunciationEmbedding` is `Codable`, so applications can choose their own persistence format and lifecycle.
FluidAudio does not own user labels, files, or dictionary storage.

## Matching Several Custom Terms

Batch all prototypes in one call. Do not invoke `bestMatch` in an application-side loop.

```swift
let matches = PronunciationEmbeddingMatcher.bestMatches(
    prototypes: prototypes,
    in: features
)

for (label, match) in zip(labels, matches) {
    guard let match else { continue }
    guard match.score >= PronunciationCustomizationDefaults.acceptanceThreshold else { continue }

    let startTime = Double(match.frameRange.lowerBound) * features.frameDuration
    let endTime = Double(match.frameRange.upperBound) * features.frameDuration
    let words = WordAudioChunkExtractor.words(from: result.tokenTimings ?? [])
    let indices = WordAudioChunkExtractor.substantiallyOverlappingWordIndices(
        in: words,
        startTime: startTime,
        endTime: endTime
    )

    // The application decides whether and how to replace `indices` with `label`.
}
```

The current exploratory defaults are:

| Setting | Default | Meaning |
|---|---:|---|
| Acceptance threshold | `0.70` | Minimum cosine score for an accepted proposal |
| Word overlap ratio | `0.50` | A decoded word must be mostly inside the acoustic match |

Applications should validate thresholds with positive and hard-negative recordings before shipping.

## Complexity and Measured Latency

Matching is linear in the number of prototypes, not quadratic. Candidate window embeddings are built once per
duration and compared in batches with Accelerate.

Release measurements on an Apple M5 Max using real Parakeet v2 encoder output:

| Audio | Prototypes | Median | p95 |
|---|---:|---:|---:|
| 1.16 seconds, fixed 640 ms duration | 100 | 0.14 ms | 0.14 ms |
| 14.8 seconds, fixed 640 ms duration | 100 | 1.06 ms | 1.09 ms |
| 14.8 seconds, mixed 320–1280 ms durations | 100 | 4.38 ms | 4.41 ms |
| 14.8 seconds, mixed durations, Accelerate forced to one thread | 100 | 4.39 ms | 4.57 ms |

Encoder capture added approximately 0.16–0.20 ms in the 14.8-second runs. One hundred 1024-value prototypes occupy
about 400 KB; the 14.8-second encoder sequence occupies about 758 KB. Lower-end Apple Silicon and Intel runtime
numbers have not yet been measured, so M5 Max results must not be presented as cross-device proof.

Run the benchmark with real audio between 1 and 15 seconds:

```bash
swift run -c release fluidaudiocli pronunciation-benchmark recording.wav \
  --runs 30 --warmup-runs 5 --varied-prototype-frames
```

The command reports 1, 10, 50, and 100 prototype timings, encoder capture cost, and memory sizes.

## Experiment Lab

The standalone macOS lab is not part of an application integration:

```bash
sh Experiments/WordAudioAlignment/run_lab.sh
```

It supports microphone enrollment, sentence recording, word-level playback, score/threshold inspection, acoustic
range playback, replacement previews, and timing logs. Its session state is intentionally not persisted.

## Limitations

- These are frozen Parakeet encoder features, not a purpose-trained triplet-loss keyword embedding model.
- Homophones such as `Claude` and `cloud` cannot be separated acoustically; context or user preference must decide.
- A lower threshold improves recall and increases false positives, especially for names resembling common words.
- Parakeet timings are quantized to 80 ms encoder frames, not sample-accurate forced alignment.
- Capture currently represents one Parakeet model pass up to 15 seconds. Longer application recordings need the same
  chunk ownership and merge discipline as the surrounding ASR pipeline.
- The API returns evidence and ranges. The application remains responsible for conflict resolution, text replacement,
  persistence, settings, and UI.

## Source and Removal Boundary

Core code lives entirely inside FluidAudio:

```text
Sources/FluidAudio/ASR/Parakeet/PronunciationCustomization/
```

The normal Parakeet path has one guarded capture call in `AsrTranscription.swift` and two default-off state fields in
`AsrManager.swift`. The lab and benchmark live under `Experiments/` and `Sources/FluidAudioCLI/Experiments/`.

To disable the feature at runtime, call `setPronunciationCustomizationEnabled(false)` and skip matching. To remove it
from a fork, delete the pronunciation customization folder, the guarded capture call/state fields, and the optional
lab/CLI experiment. No Core ML model changes are required.
