# Sherpa ONNX

`engine = "sherpa-onnx"` is an experimental native ONNX transcription
backend for benchmarking Parakeet-like models across runtimes.

V1 supports one model profile:

- `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`

The engine is behind the optional Cargo feature `sherpa-onnx`.

## Setup

Build with the feature enabled:

```bash
cargo build -p minutes-cli --features sherpa-onnx
```

Install the model and select the engine:

```bash
minutes setup --sherpa-onnx
```

By default, setup writes:

```toml
[transcription]
engine = "sherpa-onnx"
sherpa_onnx_model = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
sherpa_onnx_provider = "auto"
sherpa_onnx_num_threads = 4
sherpa_onnx_live_mode = "utterance"
```

`auto` tries a local accelerated provider first and falls back to CPU:

- Apple Silicon macOS: `coreml`, then `cpu`
- Linux/Windows: `cuda`, then `cpu`
- Other platforms: `cpu`

Use `cpu` for fair CPU-only comparisons:

```bash
minutes setup --sherpa-onnx --sherpa-onnx-provider cpu
```

## Probe

Before using the engine for saved meeting transcripts, probe an audio file:

```bash
minutes sherpa-onnx probe --audio sample.wav --flow batch
minutes sherpa-onnx probe --audio sample.wav --flow benchmark --json
```

Saved meeting transcripts require real token timestamps. If Sherpa returns text
without usable timing, the probe remains benchmark-compatible but not
batch-compatible.

## Benchmark

Run repeated decodes and report latency plus real-time factor:

```bash
minutes sherpa-onnx benchmark --audio sample.wav --runs 3
minutes sherpa-onnx benchmark --audio sample.wav --runs 3 --provider cpu --json
```

Benchmark output includes requested provider, resolved provider, fallback reason,
cold load time, per-run decode time, RTF, text length, token count, and whether
the result shape is compatible with saved meeting transcripts.

## Live and Dictation

Sherpa ONNX can be used for finalized online utterances when the binary is
built with `--features sherpa-onnx`. This uses the same warm offline recognizer
as probe/benchmark and does not require token timestamps:

```toml
[transcription]
engine = "sherpa-onnx"
sherpa_onnx_live_mode = "utterance"

[live_transcript]
backend = "inherit" # or "sherpa-onnx"

[dictation]
backend = "sherpa-onnx"
```

Whisper still produces dictation partials in this mode. Sherpa replaces only
the final text for each utterance. If the Sherpa feature, model, or provider is
unavailable during live transcript or dictation, Minutes disables Sherpa for
that session, warns once for that source, and falls back to Whisper.

## Scope

V1 supports saved-audio batch/probe/benchmark plus finalized live/dictation
utterances. True streaming partials are intentionally deferred: they should use
Sherpa's `OnlineRecognizer` with a streaming model profile, such as
`sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`, not the offline
Nemo Parakeet TDT package above.
