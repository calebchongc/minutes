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

## Scope

This first slice is batch/probe/benchmark only. Live transcript, dictation, and
desktop settings UI support are intentionally deferred until the ONNX output
shape and performance are proven on real recordings.
