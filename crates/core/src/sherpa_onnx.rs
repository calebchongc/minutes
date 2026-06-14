#[cfg(feature = "sherpa-onnx")]
use crate::config::DEFAULT_SHERPA_ONNX_NUM_THREADS;
use crate::config::{Config, DEFAULT_SHERPA_ONNX_MODEL, DEFAULT_SHERPA_ONNX_PROVIDER};
use crate::error::TranscribeError;
#[cfg(any(feature = "sherpa-onnx", test))]
use crate::transcribe::TimedTranscriptSegment;
use crate::transcribe::TranscribeResult;
#[cfg(feature = "sherpa-onnx")]
use crate::transcribe::{load_audio_samples, transcript_from_timed_segments, FilterStats};
use serde::Serialize;
use std::path::{Path, PathBuf};
#[cfg(feature = "sherpa-onnx")]
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "sherpa-onnx")]
use std::time::Instant;

pub const SHERPA_ONNX_MODEL_ARCHIVE_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2";

const ENCODER_FILE: &str = "encoder.int8.onnx";
const DECODER_FILE: &str = "decoder.int8.onnx";
const JOINER_FILE: &str = "joiner.int8.onnx";
const TOKENS_FILE: &str = "tokens.txt";
const NEMO_TRANSDUCER_MODEL_TYPE: &str = "nemo_transducer";
#[cfg(any(feature = "sherpa-onnx", test))]
const MAX_SEGMENT_SECS: f64 = 30.0;
#[cfg(any(feature = "sherpa-onnx", test))]
const SHERPA_ONNX_UTTERANCE_MIN_SAMPLES: usize = 16_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SherpaOnnxModelProfile {
    pub model: String,
    pub model_dir: PathBuf,
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub joiner: PathBuf,
    pub tokens: PathBuf,
    pub model_type: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderPlan {
    pub requested_provider: String,
    pub candidates: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SherpaOnnxProbeStats {
    pub cold_load_ms: Option<u64>,
    pub inference_ms: u64,
    pub rtf: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SherpaOnnxProbeReport {
    pub engine: String,
    pub model: String,
    pub model_dir: String,
    pub audio_path: String,
    pub requested_provider: String,
    pub resolved_provider: String,
    pub fallback_reason: Option<String>,
    pub num_threads: i32,
    pub audio_duration_secs: f64,
    pub text_chars: usize,
    pub text_preview: String,
    pub tokens: usize,
    pub token_timestamps: usize,
    pub token_durations: usize,
    pub timed_segments: usize,
    pub batch_compatible: bool,
    pub benchmark_compatible: bool,
    pub notes: Vec<String>,
    pub stats: SherpaOnnxProbeStats,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SherpaOnnxBenchmarkRun {
    pub run: usize,
    pub elapsed_ms: u64,
    pub rtf: f64,
    pub text_chars: usize,
    pub tokens: usize,
    pub token_timestamps: usize,
    pub timed_segments: usize,
    pub batch_compatible: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SherpaOnnxBenchmarkReport {
    pub engine: String,
    pub model: String,
    pub model_dir: String,
    pub requested_provider: String,
    pub resolved_provider: String,
    pub fallback_reason: Option<String>,
    pub num_threads: i32,
    pub audio_path: String,
    pub audio_duration_secs: f64,
    pub cold_load_ms: Option<u64>,
    pub runs: Vec<SherpaOnnxBenchmarkRun>,
    pub avg_elapsed_ms: f64,
    pub avg_rtf: f64,
    pub batch_compatible: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SherpaOnnxUtterance {
    pub text: String,
    pub duration_secs: f64,
}

#[cfg(any(feature = "sherpa-onnx", test))]
#[derive(Debug, Clone, PartialEq)]
struct DecodeResult {
    text: String,
    tokens: Vec<String>,
    timestamps: Option<Vec<f32>>,
    durations: Option<Vec<f32>>,
}

#[cfg(feature = "sherpa-onnx")]
#[derive(Debug, Clone, PartialEq)]
struct DecodeOutcome {
    result: DecodeResult,
    requested_provider: String,
    resolved_provider: String,
    fallback_reason: Option<String>,
    cold_load_ms: Option<u64>,
    inference_ms: u64,
}

pub fn model_archive_url(model: &str) -> Option<&'static str> {
    if model == DEFAULT_SHERPA_ONNX_MODEL {
        Some(SHERPA_ONNX_MODEL_ARCHIVE_URL)
    } else {
        None
    }
}

pub fn required_model_files(model: &str) -> Result<Vec<&'static str>, String> {
    if model != DEFAULT_SHERPA_ONNX_MODEL {
        return Err(format!(
            "unknown sherpa-onnx model `{model}`; v1 supports `{DEFAULT_SHERPA_ONNX_MODEL}`"
        ));
    }
    Ok(vec![ENCODER_FILE, DECODER_FILE, JOINER_FILE, TOKENS_FILE])
}

pub fn resolve_model_dir(config: &Config) -> PathBuf {
    config
        .transcription
        .sherpa_onnx_model_dir
        .clone()
        .unwrap_or_else(|| {
            config
                .transcription
                .model_path
                .join("sherpa-onnx")
                .join(&config.transcription.sherpa_onnx_model)
        })
}

pub fn validate_model_dir(model: &str, model_dir: &Path) -> Result<(), String> {
    for file in required_model_files(model)? {
        let path = model_dir.join(file);
        if !path.is_file() {
            return Err(format!(
                "missing Sherpa ONNX model file: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

pub fn resolve_model_profile(config: &Config) -> Result<SherpaOnnxModelProfile, String> {
    let model = config.transcription.sherpa_onnx_model.trim();
    if model != DEFAULT_SHERPA_ONNX_MODEL {
        return Err(format!(
            "unknown sherpa-onnx model `{model}`; v1 supports `{DEFAULT_SHERPA_ONNX_MODEL}`"
        ));
    }
    let model_dir = resolve_model_dir(config);
    validate_model_dir(model, &model_dir)?;
    Ok(SherpaOnnxModelProfile {
        model: model.to_string(),
        encoder: model_dir.join(ENCODER_FILE),
        decoder: model_dir.join(DECODER_FILE),
        joiner: model_dir.join(JOINER_FILE),
        tokens: model_dir.join(TOKENS_FILE),
        model_dir,
        model_type: NEMO_TRANSDUCER_MODEL_TYPE,
    })
}

pub fn provider_plan(requested: &str) -> Result<ProviderPlan, String> {
    let requested = normalize_provider(requested);
    let candidates = match requested.as_str() {
        "auto" => auto_provider_candidates(),
        "cpu" | "coreml" | "cuda" => vec![requested.clone()],
        other => {
            return Err(format!(
                "unknown sherpa-onnx provider `{other}`; expected auto, cpu, coreml, or cuda"
            ))
        }
    };
    Ok(ProviderPlan {
        requested_provider: requested,
        candidates,
    })
}

fn normalize_provider(provider: &str) -> String {
    let provider = provider.trim();
    if provider.is_empty() {
        DEFAULT_SHERPA_ONNX_PROVIDER.into()
    } else {
        provider.to_ascii_lowercase()
    }
}

fn auto_provider_candidates() -> Vec<String> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        vec!["coreml".into(), "cpu".into()]
    }
    #[cfg(all(
        not(all(target_os = "macos", target_arch = "aarch64")),
        any(target_os = "linux", target_os = "windows")
    ))]
    {
        vec!["cuda".into(), "cpu".into()]
    }
    #[cfg(all(
        not(all(target_os = "macos", target_arch = "aarch64")),
        not(any(target_os = "linux", target_os = "windows"))
    ))]
    {
        vec!["cpu".into()]
    }
}

pub fn transcribe(audio_path: &Path, config: &Config) -> Result<TranscribeResult, TranscribeError> {
    transcribe_impl(audio_path, config)
}

pub fn probe(audio_path: &Path, config: &Config) -> Result<SherpaOnnxProbeReport, TranscribeError> {
    probe_impl(audio_path, config)
}

pub fn benchmark(
    audio_path: &Path,
    config: &Config,
    runs: usize,
) -> Result<SherpaOnnxBenchmarkReport, TranscribeError> {
    benchmark_impl(audio_path, config, runs)
}

pub fn transcribe_utterance(
    samples: &[f32],
    config: &Config,
) -> Result<Option<SherpaOnnxUtterance>, TranscribeError> {
    transcribe_utterance_impl(samples, config)
}

pub fn reset_warm_cache() {
    #[cfg(feature = "sherpa-onnx")]
    {
        if let Some(cache) = RECOGNIZER_CACHE.get() {
            if let Ok(mut cache) = cache.lock() {
                *cache = None;
            }
        }
    }
}

#[cfg(not(feature = "sherpa-onnx"))]
fn transcribe_utterance_impl(
    _samples: &[f32],
    _config: &Config,
) -> Result<Option<SherpaOnnxUtterance>, TranscribeError> {
    Err(TranscribeError::EngineNotAvailable("sherpa-onnx".into()))
}

#[cfg(not(feature = "sherpa-onnx"))]
fn transcribe_impl(
    _audio_path: &Path,
    _config: &Config,
) -> Result<TranscribeResult, TranscribeError> {
    Err(TranscribeError::EngineNotAvailable("sherpa-onnx".into()))
}

#[cfg(not(feature = "sherpa-onnx"))]
fn probe_impl(
    _audio_path: &Path,
    _config: &Config,
) -> Result<SherpaOnnxProbeReport, TranscribeError> {
    Err(TranscribeError::EngineNotAvailable("sherpa-onnx".into()))
}

#[cfg(not(feature = "sherpa-onnx"))]
fn benchmark_impl(
    _audio_path: &Path,
    _config: &Config,
    _runs: usize,
) -> Result<SherpaOnnxBenchmarkReport, TranscribeError> {
    Err(TranscribeError::EngineNotAvailable("sherpa-onnx".into()))
}

#[cfg(feature = "sherpa-onnx")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct RecognizerKey {
    model: String,
    model_dir: PathBuf,
    requested_provider: String,
    num_threads: i32,
}

#[cfg(feature = "sherpa-onnx")]
struct CachedRecognizer {
    key: RecognizerKey,
    resolved_provider: String,
    fallback_reason: Option<String>,
    recognizer: ::sherpa_onnx::OfflineRecognizer,
}

#[cfg(feature = "sherpa-onnx")]
static RECOGNIZER_CACHE: OnceLock<Mutex<Option<CachedRecognizer>>> = OnceLock::new();

#[cfg(feature = "sherpa-onnx")]
fn recognizer_cache() -> &'static Mutex<Option<CachedRecognizer>> {
    RECOGNIZER_CACHE.get_or_init(|| Mutex::new(None))
}

#[cfg(feature = "sherpa-onnx")]
fn transcribe_utterance_impl(
    samples: &[f32],
    config: &Config,
) -> Result<Option<SherpaOnnxUtterance>, TranscribeError> {
    if samples.is_empty() {
        return Err(TranscribeError::EmptyAudio);
    }
    if samples.len() < SHERPA_ONNX_UTTERANCE_MIN_SAMPLES {
        return Ok(None);
    }

    let duration_secs = samples.len() as f64 / 16_000.0;
    let outcome = decode_samples(samples, config)?;
    let utterance = utterance_from_decode_result(&outcome.result, duration_secs);
    if let Some(utterance) = &utterance {
        tracing::info!(
            engine = "sherpa-onnx",
            requested_provider = outcome.requested_provider,
            resolved_provider = outcome.resolved_provider,
            inference_ms = outcome.inference_ms,
            chars = utterance.text.chars().count(),
            "sherpa-onnx finalized utterance complete"
        );
    }
    Ok(utterance)
}

#[cfg(feature = "sherpa-onnx")]
fn transcribe_impl(
    audio_path: &Path,
    config: &Config,
) -> Result<TranscribeResult, TranscribeError> {
    let (samples, audio_duration_secs) = load_checked_samples(audio_path)?;
    let outcome = decode_samples(&samples, config)?;
    let timed_segments = timed_segments_from_decode_result(&outcome.result).map_err(|error| {
        TranscribeError::TranscriptionFailed(format!(
            "sherpa-onnx output is not compatible with saved meeting transcripts: {error}"
        ))
    })?;
    let (text, cleanup_stats) = transcript_from_timed_segments(&timed_segments, config)?;
    let mut stats = FilterStats {
        audio_duration_secs,
        samples_after_silence_strip: samples.len(),
        raw_segments: cleanup_stats.raw_segments,
        after_no_speech_filter: cleanup_stats.raw_segments,
        after_dedup: cleanup_stats.after_dedup,
        after_interleaved: cleanup_stats.after_interleaved,
        after_script_filter: cleanup_stats.after_script_filter,
        after_noise_markers: cleanup_stats.after_noise_markers,
        after_trailing_trim: cleanup_stats.after_trailing_trim,
        final_words: text.split_whitespace().count(),
        ..FilterStats::default()
    };
    stats.after_no_speech_filter = stats.raw_segments;

    tracing::info!(
        engine = "sherpa-onnx",
        requested_provider = outcome.requested_provider,
        resolved_provider = outcome.resolved_provider,
        inference_ms = outcome.inference_ms,
        words = stats.final_words,
        diagnosis = stats.diagnosis(),
        "sherpa-onnx transcription complete"
    );

    Ok(TranscribeResult { text, stats })
}

#[cfg(feature = "sherpa-onnx")]
fn probe_impl(
    audio_path: &Path,
    config: &Config,
) -> Result<SherpaOnnxProbeReport, TranscribeError> {
    let (samples, audio_duration_secs) = load_checked_samples(audio_path)?;
    let outcome = decode_samples(&samples, config)?;
    Ok(probe_report_from_outcome(
        audio_path,
        config,
        audio_duration_secs,
        outcome,
    ))
}

#[cfg(feature = "sherpa-onnx")]
fn benchmark_impl(
    audio_path: &Path,
    config: &Config,
    runs: usize,
) -> Result<SherpaOnnxBenchmarkReport, TranscribeError> {
    let runs = runs.max(1);
    let (samples, audio_duration_secs) = load_checked_samples(audio_path)?;
    reset_warm_cache();

    let mut benchmark_runs = Vec::with_capacity(runs);
    let mut cold_load_ms = None;
    let mut resolved_provider = String::new();
    let mut requested_provider = String::new();
    let mut fallback_reason = None;
    let mut batch_compatible = false;

    for run_index in 0..runs {
        let outcome = decode_samples(&samples, config)?;
        if run_index == 0 {
            cold_load_ms = outcome.cold_load_ms;
            requested_provider = outcome.requested_provider.clone();
            resolved_provider = outcome.resolved_provider.clone();
            fallback_reason = outcome.fallback_reason.clone();
        }
        let timed_segments = timed_segments_from_decode_result(&outcome.result).unwrap_or_default();
        let current_batch_compatible = !timed_segments.is_empty();
        batch_compatible |= current_batch_compatible;
        benchmark_runs.push(SherpaOnnxBenchmarkRun {
            run: run_index + 1,
            elapsed_ms: outcome.inference_ms,
            rtf: rtf(outcome.inference_ms, audio_duration_secs),
            text_chars: outcome.result.text.chars().count(),
            tokens: outcome.result.tokens.len(),
            token_timestamps: outcome.result.timestamps.as_ref().map_or(0, Vec::len),
            timed_segments: timed_segments.len(),
            batch_compatible: current_batch_compatible,
        });
    }

    let profile = resolve_model_profile(config).map_err(TranscribeError::TranscriptionFailed)?;
    let avg_elapsed_ms = benchmark_runs
        .iter()
        .map(|run| run.elapsed_ms as f64)
        .sum::<f64>()
        / runs as f64;
    let avg_rtf = benchmark_runs.iter().map(|run| run.rtf).sum::<f64>() / runs as f64;
    Ok(SherpaOnnxBenchmarkReport {
        engine: "sherpa-onnx".into(),
        model: profile.model,
        model_dir: profile.model_dir.display().to_string(),
        requested_provider,
        resolved_provider,
        fallback_reason,
        num_threads: effective_num_threads(config),
        audio_path: audio_path.display().to_string(),
        audio_duration_secs,
        cold_load_ms,
        runs: benchmark_runs,
        avg_elapsed_ms,
        avg_rtf,
        batch_compatible,
    })
}

#[cfg(feature = "sherpa-onnx")]
fn load_checked_samples(audio_path: &Path) -> Result<(Vec<f32>, f64), TranscribeError> {
    let samples = load_audio_samples(audio_path)?;
    if samples.is_empty() {
        return Err(TranscribeError::EmptyAudio);
    }
    let audio_duration_secs = samples.len() as f64 / 16_000.0;
    Ok((samples, audio_duration_secs))
}

#[cfg(feature = "sherpa-onnx")]
fn decode_samples(samples: &[f32], config: &Config) -> Result<DecodeOutcome, TranscribeError> {
    let profile = resolve_model_profile(config).map_err(TranscribeError::TranscriptionFailed)?;
    let provider_plan = provider_plan(&config.transcription.sherpa_onnx_provider)
        .map_err(TranscribeError::TranscriptionFailed)?;
    let key = RecognizerKey {
        model: profile.model.clone(),
        model_dir: profile.model_dir.clone(),
        requested_provider: provider_plan.requested_provider.clone(),
        num_threads: effective_num_threads(config),
    };

    let mut cache = recognizer_cache().lock().map_err(|_| {
        TranscribeError::TranscriptionFailed("sherpa-onnx recognizer cache lock poisoned".into())
    })?;

    let mut cold_load_ms = None;
    let needs_reload = cache.as_ref().map_or(true, |cached| cached.key != key);
    if needs_reload {
        let started = Instant::now();
        let cached = create_cached_recognizer(&profile, &provider_plan, key.clone())?;
        cold_load_ms = Some(started.elapsed().as_millis() as u64);
        *cache = Some(cached);
    }

    let cached = cache.as_ref().ok_or_else(|| {
        TranscribeError::TranscriptionFailed("sherpa-onnx recognizer missing".into())
    })?;
    let started = Instant::now();
    let stream = cached.recognizer.create_stream();
    stream.accept_waveform(16_000, samples);
    cached.recognizer.decode(&stream);
    let result = stream.get_result().ok_or_else(|| {
        TranscribeError::TranscriptionFailed("sherpa-onnx recognizer returned no result".into())
    })?;
    let inference_ms = started.elapsed().as_millis() as u64;

    Ok(DecodeOutcome {
        result: DecodeResult {
            text: result.text,
            tokens: result.tokens,
            timestamps: result.timestamps,
            durations: result.durations,
        },
        requested_provider: provider_plan.requested_provider,
        resolved_provider: cached.resolved_provider.clone(),
        fallback_reason: cached.fallback_reason.clone(),
        cold_load_ms,
        inference_ms,
    })
}

#[cfg(feature = "sherpa-onnx")]
fn create_cached_recognizer(
    profile: &SherpaOnnxModelProfile,
    provider_plan: &ProviderPlan,
    key: RecognizerKey,
) -> Result<CachedRecognizer, TranscribeError> {
    let mut failures = Vec::new();
    for provider in &provider_plan.candidates {
        let config = recognizer_config(profile, provider, key.num_threads);
        if let Some(recognizer) = ::sherpa_onnx::OfflineRecognizer::create(&config) {
            let fallback_reason = if provider != &provider_plan.candidates[0] {
                Some(format!(
                    "provider `{}` failed; using `{}` ({})",
                    provider_plan.candidates[0],
                    provider,
                    failures.join("; ")
                ))
            } else {
                None
            };
            return Ok(CachedRecognizer {
                key,
                resolved_provider: provider.clone(),
                fallback_reason,
                recognizer,
            });
        }
        failures.push(format!("`{provider}` recognizer creation returned null"));
        if provider_plan.requested_provider != "auto" {
            break;
        }
    }

    Err(TranscribeError::TranscriptionFailed(format!(
        "failed to create sherpa-onnx recognizer for provider `{}`: {}",
        provider_plan.requested_provider,
        failures.join("; ")
    )))
}

#[cfg(feature = "sherpa-onnx")]
fn recognizer_config(
    profile: &SherpaOnnxModelProfile,
    provider: &str,
    num_threads: i32,
) -> ::sherpa_onnx::OfflineRecognizerConfig {
    let mut config = ::sherpa_onnx::OfflineRecognizerConfig::default();
    config.model_config.transducer = ::sherpa_onnx::OfflineTransducerModelConfig {
        encoder: Some(profile.encoder.display().to_string()),
        decoder: Some(profile.decoder.display().to_string()),
        joiner: Some(profile.joiner.display().to_string()),
    };
    config.model_config.tokens = Some(profile.tokens.display().to_string());
    config.model_config.model_type = Some(profile.model_type.into());
    config.model_config.provider = Some(provider.into());
    config.model_config.num_threads = num_threads;
    config
}

#[cfg(feature = "sherpa-onnx")]
fn effective_num_threads(config: &Config) -> i32 {
    if config.transcription.sherpa_onnx_num_threads <= 0 {
        DEFAULT_SHERPA_ONNX_NUM_THREADS
    } else {
        config.transcription.sherpa_onnx_num_threads
    }
}

#[cfg(feature = "sherpa-onnx")]
fn probe_report_from_outcome(
    audio_path: &Path,
    config: &Config,
    audio_duration_secs: f64,
    outcome: DecodeOutcome,
) -> SherpaOnnxProbeReport {
    let profile = resolve_model_profile(config).unwrap_or_else(|_| SherpaOnnxModelProfile {
        model: config.transcription.sherpa_onnx_model.clone(),
        model_dir: resolve_model_dir(config),
        encoder: PathBuf::new(),
        decoder: PathBuf::new(),
        joiner: PathBuf::new(),
        tokens: PathBuf::new(),
        model_type: NEMO_TRANSDUCER_MODEL_TYPE,
    });
    let timed_segments = timed_segments_from_decode_result(&outcome.result).unwrap_or_default();
    let text_present = !outcome.result.text.trim().is_empty()
        || outcome.result.tokens.iter().any(|t| !t.trim().is_empty());
    let batch_compatible = !timed_segments.is_empty();
    let mut notes = Vec::new();
    if batch_compatible {
        notes.push("Saved meeting transcripts can use this output shape: Sherpa returned usable token timing.".into());
    } else if text_present {
        notes.push("Sherpa returned transcript text but no usable token timing; benchmarking can use it, but saved meeting transcripts require real timestamps.".into());
    } else {
        notes.push("Sherpa did not return transcript text in this run.".into());
    }
    if outcome.resolved_provider != outcome.requested_provider
        && outcome.requested_provider != "auto"
    {
        notes.push(format!(
            "Requested provider `{}` resolved to `{}`.",
            outcome.requested_provider, outcome.resolved_provider
        ));
    }
    if let Some(reason) = &outcome.fallback_reason {
        notes.push(reason.clone());
    }

    SherpaOnnxProbeReport {
        engine: "sherpa-onnx".into(),
        model: profile.model,
        model_dir: profile.model_dir.display().to_string(),
        audio_path: audio_path.display().to_string(),
        requested_provider: outcome.requested_provider,
        resolved_provider: outcome.resolved_provider,
        fallback_reason: outcome.fallback_reason,
        num_threads: effective_num_threads(config),
        audio_duration_secs,
        text_chars: outcome.result.text.chars().count(),
        text_preview: char_preview(&outcome.result.text, 200),
        tokens: outcome.result.tokens.len(),
        token_timestamps: outcome.result.timestamps.as_ref().map_or(0, Vec::len),
        token_durations: outcome.result.durations.as_ref().map_or(0, Vec::len),
        timed_segments: timed_segments.len(),
        batch_compatible,
        benchmark_compatible: text_present,
        notes,
        stats: SherpaOnnxProbeStats {
            cold_load_ms: outcome.cold_load_ms,
            inference_ms: outcome.inference_ms,
            rtf: rtf(outcome.inference_ms, audio_duration_secs),
        },
    }
}

#[cfg(any(feature = "sherpa-onnx", test))]
fn timed_segments_from_decode_result(
    result: &DecodeResult,
) -> Result<Vec<TimedTranscriptSegment>, String> {
    let timestamps = result
        .timestamps
        .as_ref()
        .ok_or_else(|| "missing token timestamps".to_string())?;
    if result.tokens.is_empty() {
        return Err("missing tokens".into());
    }
    if timestamps.len() < result.tokens.len() {
        return Err(format!(
            "timestamp count {} is smaller than token count {}",
            timestamps.len(),
            result.tokens.len()
        ));
    }

    let mut timed_tokens: Vec<TimedToken> = Vec::new();
    for (index, token) in result.tokens.iter().enumerate() {
        let text = normalize_token_text(token);
        if text.trim().is_empty() {
            continue;
        }
        let start = timestamps[index] as f64;
        if !start.is_finite() || start < 0.0 {
            return Err(format!(
                "token timestamp at index {index} is invalid: {start}"
            ));
        }
        if let Some(previous) = timed_tokens.last() {
            if start < previous.start_secs {
                return Err("token timestamps are not monotonic".into());
            }
        }
        let end = result
            .durations
            .as_ref()
            .and_then(|durations| durations.get(index))
            .map(|duration| start + *duration as f64)
            .filter(|end| end.is_finite() && *end > start)
            .unwrap_or(start);
        timed_tokens.push(TimedToken {
            start_secs: start,
            end_secs: end,
            text,
        });
    }

    if timed_tokens.is_empty() {
        return Err("tokens normalize to empty text".into());
    }

    let mut segments = Vec::new();
    let mut segment_start = timed_tokens[0].start_secs;
    let mut segment_end = timed_tokens[0].end_secs;
    let mut text_parts = Vec::new();

    for token in timed_tokens {
        if text_parts.is_empty() {
            segment_start = token.start_secs;
        }
        segment_end = segment_end.max(token.end_secs);
        text_parts.push(token.text);
        let joined = normalize_joined_text(&text_parts.join(""));
        let has_boundary = joined.ends_with('.') || joined.ends_with('?') || joined.ends_with('!');
        let long_enough = segment_end - segment_start >= MAX_SEGMENT_SECS;
        if (has_boundary || long_enough) && segment_end > segment_start && !joined.is_empty() {
            segments.push(TimedTranscriptSegment {
                start_secs: segment_start,
                end_secs: segment_end,
                text: joined,
            });
            text_parts.clear();
        }
    }

    if !text_parts.is_empty() {
        let joined = normalize_joined_text(&text_parts.join(""));
        if segment_end > segment_start && !joined.is_empty() {
            segments.push(TimedTranscriptSegment {
                start_secs: segment_start,
                end_secs: segment_end,
                text: joined,
            });
        }
    }

    if segments.is_empty() {
        return Err("token timestamps did not produce positive-duration segments".into());
    }
    Ok(segments)
}

#[cfg(any(feature = "sherpa-onnx", test))]
fn utterance_from_decode_result(
    result: &DecodeResult,
    duration_secs: f64,
) -> Option<SherpaOnnxUtterance> {
    let text = text_from_decode_result(result)?;
    Some(SherpaOnnxUtterance {
        text,
        duration_secs,
    })
}

#[cfg(any(feature = "sherpa-onnx", test))]
fn text_from_decode_result(result: &DecodeResult) -> Option<String> {
    let direct = normalize_joined_text(&result.text);
    if !direct.is_empty() {
        return Some(direct);
    }

    let joined_tokens = result
        .tokens
        .iter()
        .map(|token| normalize_token_text(token))
        .filter(|token| !token.trim().is_empty())
        .collect::<Vec<_>>()
        .join("");
    let joined_tokens = normalize_joined_text(&joined_tokens);
    if joined_tokens.is_empty() {
        None
    } else {
        Some(joined_tokens)
    }
}

#[cfg(any(feature = "sherpa-onnx", test))]
#[derive(Debug, Clone)]
struct TimedToken {
    start_secs: f64,
    end_secs: f64,
    text: String,
}

#[cfg(any(feature = "sherpa-onnx", test))]
fn normalize_token_text(token: &str) -> String {
    let trimmed = token.trim();
    if trimmed.starts_with('<') && trimmed.ends_with('>') {
        return String::new();
    }
    token
        .replace('\u{2581}', " ")
        .replace('\u{0120}', " ")
        .replace("##", "")
}

#[cfg(any(feature = "sherpa-onnx", test))]
fn normalize_joined_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(feature = "sherpa-onnx")]
fn char_preview(text: &str, limit: usize) -> String {
    text.chars().take(limit).collect()
}

#[cfg(feature = "sherpa-onnx")]
fn rtf(elapsed_ms: u64, audio_duration_secs: f64) -> f64 {
    if audio_duration_secs <= 0.0 {
        0.0
    } else {
        elapsed_ms as f64 / 1000.0 / audio_duration_secs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TranscriptionConfig;

    #[test]
    fn provider_plan_accepts_explicit_providers() {
        assert_eq!(provider_plan("cpu").unwrap().candidates, vec!["cpu"]);
        assert_eq!(provider_plan("coreml").unwrap().candidates, vec!["coreml"]);
        assert_eq!(provider_plan("cuda").unwrap().candidates, vec!["cuda"]);
    }

    #[test]
    fn provider_plan_auto_has_cpu_fallback() {
        let plan = provider_plan("auto").unwrap();
        assert_eq!(plan.requested_provider, "auto");
        assert!(plan.candidates.iter().any(|provider| provider == "cpu"));
    }

    #[test]
    fn provider_plan_rejects_unknown_provider() {
        let error = provider_plan("directml").unwrap_err();
        assert!(error.contains("unknown sherpa-onnx provider"));
    }

    #[test]
    fn default_model_dir_uses_transcription_model_path() {
        let mut config = Config::default();
        config.transcription = TranscriptionConfig::default();
        config.transcription.model_path = PathBuf::from("/tmp/minutes-models");
        assert_eq!(
            resolve_model_dir(&config),
            PathBuf::from("/tmp/minutes-models")
                .join("sherpa-onnx")
                .join(DEFAULT_SHERPA_ONNX_MODEL)
        );
    }

    #[test]
    fn timestamped_tokens_convert_to_timed_segments() {
        let result = DecodeResult {
            text: "hello world.".into(),
            tokens: vec![" hello".into(), " world".into(), ".".into()],
            timestamps: Some(vec![0.0, 0.5, 1.0]),
            durations: Some(vec![0.2, 0.2, 0.2]),
        };

        let segments = timed_segments_from_decode_result(&result).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_secs, 0.0);
        assert!(segments[0].end_secs > segments[0].start_secs);
        assert_eq!(segments[0].text, "hello world.");
    }

    #[test]
    fn text_only_result_is_not_batch_compatible() {
        let result = DecodeResult {
            text: "hello world".into(),
            tokens: vec!["hello".into(), "world".into()],
            timestamps: None,
            durations: None,
        };

        let error = timed_segments_from_decode_result(&result).unwrap_err();
        assert!(error.contains("missing token timestamps"));
    }

    #[test]
    fn text_only_result_is_accepted_for_finalized_utterance() {
        let result = DecodeResult {
            text: " hello   world ".into(),
            tokens: Vec::new(),
            timestamps: None,
            durations: None,
        };

        let utterance = utterance_from_decode_result(&result, 1.25).unwrap();
        assert_eq!(utterance.text, "hello world");
        assert_eq!(utterance.duration_secs, 1.25);
    }

    #[test]
    fn finalized_utterance_can_fall_back_to_tokens() {
        let result = DecodeResult {
            text: String::new(),
            tokens: vec!["\u{2581}hello".into(), "\u{2581}world".into()],
            timestamps: None,
            durations: None,
        };

        let utterance = utterance_from_decode_result(&result, 2.0).unwrap();
        assert_eq!(utterance.text, "hello world");
        assert_eq!(utterance.duration_secs, 2.0);
    }

    #[test]
    fn empty_result_is_not_a_finalized_utterance() {
        let result = DecodeResult {
            text: "   ".into(),
            tokens: vec!["<blk>".into()],
            timestamps: None,
            durations: None,
        };

        assert_eq!(utterance_from_decode_result(&result, 1.0), None);
    }

    #[test]
    #[cfg(not(feature = "sherpa-onnx"))]
    fn utterance_returns_engine_not_available_without_feature() {
        let samples = vec![0.0; SHERPA_ONNX_UTTERANCE_MIN_SAMPLES];
        let error = transcribe_utterance(&samples, &Config::default()).unwrap_err();
        assert!(matches!(
            error,
            TranscribeError::EngineNotAvailable(engine) if engine == "sherpa-onnx"
        ));
    }

    #[test]
    fn invalid_timestamps_are_rejected() {
        let result = DecodeResult {
            text: "hello world".into(),
            tokens: vec!["hello".into(), "world".into()],
            timestamps: Some(vec![1.0, 0.5]),
            durations: Some(vec![0.2, 0.2]),
        };

        let error = timed_segments_from_decode_result(&result).unwrap_err();
        assert!(error.contains("not monotonic"));
    }
}
