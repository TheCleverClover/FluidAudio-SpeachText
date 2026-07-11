# Word Audio Alignment Probe

This removable experiment verifies that Parakeet token timings can map a selected transcribed word back to the same
16 kHz samples sent to ASR. It does not change inference or transcript text.

## Run the UI

```bash
sh Experiments/WordAudioAlignment/run_lab.sh
```

The lab records one sentence, transcribes it with Parakeet v2, shows the complete recording, and creates one playable
row per word. The Raw / +80 ms / +120 ms control changes listening context without changing the reported model
boundary.

The Embedding experiment is session-only and intentionally diagnostic:

1. Enter the intended spelling.
2. Record the isolated word 3–10 times, or use existing isolated audio files.
3. Record/open a sentence containing the word.
4. Inspect the cosine score, matched time range, overlapping decoded words, and replacement preview.
5. Play the matched acoustic window to verify what would be replaced.

Matching uses mean-pooled, L2-normalized Parakeet encoder frames. It does not compare the typed label against the
transcript. Encoder capture is opt-in and disabled for every normal `AsrManager` unless explicitly enabled.

The reusable implementation lives under
`Sources/FluidAudio/ASR/Parakeet/PronunciationCustomization/`. See
`Documentation/ASR/PronunciationCustomization.md` for the integration boundary, API lifecycle, benchmark results,
and limitations.

Logs use `LAB_ENROLL`, `LAB_SENTENCE`, and `LAB_MATCH`. They include ASR time, encoder capture time/bytes, frame
counts, match score, threshold, matched timestamps, overlapping words, and acceptance decision.

## Run the CLI probe

```bash
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
  xcrun swift run fluidaudiocli word-audio-probe recording.wav \
  --word Bharatwaj \
  --padding-ms 120 \
  --output Experiments/WordAudioAlignment/output/bharatwaj.wav \
  --play
```

Use `--occurrence 2` to select a later occurrence. Use `--padding-ms 0` to hear the raw model timing boundary.

Benchmark 1, 10, 50, and 100 prototypes with real Parakeet encoder features:

```bash
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
  xcrun swift run -c release fluidaudiocli pronunciation-benchmark recording.wav \
  --runs 30 --warmup-runs 5 --varied-prototype-frames
```

## Evidence

The command logs:

- the full transcript;
- every reconstructed word with start/end time and confidence;
- the selected raw and padded time ranges;
- the exact 16 kHz sample range;
- transcription, extraction, and total timing;
- the playable WAV path.

`WORD_CHUNK_TIMING extractionUs` is the added alignment/slicing cost. Model loading is included in `totalMs`, but not
in `transcriptionMs`.

## Current limitation

Parakeet timings are emitted on 80 ms encoder-frame boundaries. The extracted range is therefore a model alignment,
not sample-accurate forced alignment. Listening tests determine whether that resolution is sufficient before the
embedding experiment continues.
