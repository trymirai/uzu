#![cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::Deserialize;
use uzu::session::{
    TtsSession,
    config::{TextDecoderFollowupStrategy, TextSamplingConfig, TtsRunConfig, TtsSessionOptions},
    types::Input,
};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

const FISHAUDIO_F32_GREEDY_FIXTURE_PATH: &str = "tests/fixtures/fishaudio_s1mini_hello_lucky_greedy_tokens.json";

#[derive(Debug, Deserialize)]
struct FishAudioGreedyFixture {
    text: String,
    seed: u64,
    codebooks: usize,
    frames: usize,
    lengths: Vec<usize>,
    tokens_codebook_major: Vec<u32>,
}

fn greedy_fishaudio_session_options() -> TtsSessionOptions {
    let mut options = TtsSessionOptions::default();
    options.text_decoder.sampling = TextSamplingConfig {
        temperature: 0.0,
        top_p: 0.0,
        repetition_penalty: 1.0,
    };
    options
}

fn load_optional_model_path() -> Option<PathBuf> {
    let path = std::env::var("LALAMO_UZU_MODEL_PATH").ok().map(PathBuf::from)?;
    assert!(
        path.join("config.json").exists() && path.join("model.safetensors").exists(),
        "LALAMO_UZU_MODEL_PATH does not point to a valid export: {}",
        path.display()
    );
    Some(path)
}

fn load_optional_fishaudio_model_path() -> Option<PathBuf> {
    let path = std::env::var("LALAMO_UZU_FISHAUDIO_MODEL_PATH").ok().map(PathBuf::from)?;
    assert!(
        path.join("config.json").exists() && path.join("model.safetensors").exists(),
        "LALAMO_UZU_FISHAUDIO_MODEL_PATH does not point to a valid export: {}",
        path.display()
    );
    Some(path)
}

fn load_optional_fishaudio_model_path_f16() -> Option<PathBuf> {
    let path = std::env::var("LALAMO_UZU_FISHAUDIO_MODEL_PATH_F16").ok().map(PathBuf::from)?;
    assert!(
        path.join("config.json").exists() && path.join("model.safetensors").exists(),
        "LALAMO_UZU_FISHAUDIO_MODEL_PATH_F16 does not point to a valid export: {}",
        path.display()
    );
    Some(path)
}

fn load_optional_fishaudio_model_path_bf16() -> Option<PathBuf> {
    let path = std::env::var("LALAMO_UZU_FISHAUDIO_MODEL_PATH_BF16").ok().map(PathBuf::from)?;
    assert!(
        path.join("config.json").exists() && path.join("model.safetensors").exists(),
        "LALAMO_UZU_FISHAUDIO_MODEL_PATH_BF16 does not point to a valid export: {}",
        path.display()
    );
    Some(path)
}

fn load_fishaudio_greedy_fixture() -> Option<FishAudioGreedyFixture> {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FISHAUDIO_F32_GREEDY_FIXTURE_PATH);
    let payload = fs::read_to_string(&fixture_path).ok()?;
    serde_json::from_str::<FishAudioGreedyFixture>(&payload).ok()
}

#[test]
fn tts_session_synthesizes_audio_from_text_on_normal_export() {
    let _guard = TEST_MUTEX.lock().expect("lock");
    let Some(model_path) = load_optional_model_path() else {
        println!("Skipping TTS session test: set LALAMO_UZU_MODEL_PATH");
        return;
    };

    let session = TtsSession::new(model_path).expect("tts session");
    let pcm = session.synthesize(Input::Text("hello".to_string())).expect("synthesize");

    assert_eq!(pcm.sample_rate(), session.sample_rate());
    assert_eq!(pcm.lengths().len(), 1);
    assert_eq!(pcm.channels(), 1);
    assert!(pcm.lengths()[0] > 0, "expected non-empty waveform for non-empty text");
    assert_eq!(pcm.samples().len(), pcm.lengths()[0] * pcm.channels());
}

#[test]
fn tts_session_streaming_matches_non_streaming_audio_on_normal_export() {
    let _guard = TEST_MUTEX.lock().expect("lock");
    let Some(model_path) = load_optional_model_path() else {
        println!("Skipping streaming TTS session test: set LALAMO_UZU_MODEL_PATH");
        return;
    };

    let session = TtsSession::new(model_path).expect("tts session");
    let seed = 123_u64;
    let baseline = session.synthesize_with_seed(Input::Text("hello".to_string()), seed).expect("baseline");

    let mut streamed_samples = Vec::<f32>::new();
    let streamed = session
        .synthesize_streaming_with_seed(Input::Text("hello".to_string()), seed, 4, |chunk| {
            streamed_samples.extend_from_slice(chunk.samples());
        })
        .expect("streamed");

    assert_eq!(streamed.sample_rate(), baseline.sample_rate());
    assert_eq!(streamed.channels(), baseline.channels());
    assert_eq!(streamed.lengths(), baseline.lengths());
    assert_eq!(streamed.samples(), baseline.samples());
    assert_eq!(streamed_samples, baseline.samples());
}

#[test]
fn tts_session_fishaudio_f32_greedy_tokens_match_fixture() {
    let _guard = TEST_MUTEX.lock().expect("lock");
    let Some(model_path) = load_optional_fishaudio_model_path() else {
        println!("Skipping FishAudio f32 parity test: set LALAMO_UZU_FISHAUDIO_MODEL_PATH");
        return;
    };
    let Some(fixture) = load_fishaudio_greedy_fixture() else {
        println!(
            "Skipping FishAudio f32 parity test: missing fixture at {}",
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FISHAUDIO_F32_GREEDY_FIXTURE_PATH).display()
        );
        return;
    };

    let session = TtsSession::new_with_options(model_path, greedy_fishaudio_session_options()).expect("tts session");
    let tokens = session
        .generate_semantic_tokens_with_seed(Input::Text(fixture.text.clone()), fixture.seed)
        .expect("generate semantic tokens");

    assert_eq!(tokens.batch_size(), 1);
    assert_eq!(tokens.codebooks(), fixture.codebooks);
    assert_eq!(tokens.frames(), fixture.frames);
    assert_eq!(tokens.lengths(), fixture.lengths.as_slice());
    assert_eq!(tokens.tokens(), fixture.tokens_codebook_major.as_slice());
}

#[test]
fn tts_session_fishaudio_f32_semantic_generation_respects_frame_cap() {
    let _guard = TEST_MUTEX.lock().expect("lock");
    let Some(model_path) = load_optional_fishaudio_model_path() else {
        println!("Skipping FishAudio f32 semantic-cap test: set LALAMO_UZU_FISHAUDIO_MODEL_PATH");
        return;
    };
    let Some(fixture) = load_fishaudio_greedy_fixture() else {
        println!(
            "Skipping FishAudio f32 semantic-cap test: missing fixture at {}",
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FISHAUDIO_F32_GREEDY_FIXTURE_PATH).display()
        );
        return;
    };

    let session = TtsSession::new_with_options(model_path, greedy_fishaudio_session_options()).expect("tts session");

    let frame_cap = 8usize;
    let config = TtsRunConfig {
        max_semantic_frames: frame_cap,
        ..TtsRunConfig::default()
    };
    let tokens = session
        .generate_semantic_tokens_with_seed_and_config(Input::Text(fixture.text.clone()), fixture.seed, &config)
        .expect("generate capped semantic tokens");

    assert_eq!(tokens.batch_size(), 1);
    assert_eq!(tokens.codebooks(), fixture.codebooks);
    assert_eq!(tokens.frames(), frame_cap);
    assert_eq!(tokens.lengths(), [frame_cap]);

    let mut expected_prefix = Vec::with_capacity(fixture.codebooks * frame_cap);
    for codebook in 0..fixture.codebooks {
        let base = codebook * fixture.frames;
        expected_prefix.extend_from_slice(&fixture.tokens_codebook_major[base..base + frame_cap]);
    }
    assert_eq!(tokens.tokens(), expected_prefix.as_slice());
}

#[test]
fn tts_session_fishaudio_f32_streaming_matches_non_streaming_audio_with_windowed_context() {
    let _guard = TEST_MUTEX.lock().expect("lock");
    let Some(model_path) = load_optional_fishaudio_model_path() else {
        println!("Skipping FishAudio f32 streaming parity test: set LALAMO_UZU_FISHAUDIO_MODEL_PATH");
        return;
    };
    let Some(fixture) = load_fishaudio_greedy_fixture() else {
        println!(
            "Skipping FishAudio f32 streaming parity test: missing fixture at {}",
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FISHAUDIO_F32_GREEDY_FIXTURE_PATH).display()
        );
        return;
    };

    let session = TtsSession::new_with_options(model_path, greedy_fishaudio_session_options()).expect("tts session");

    // This cap forces decode to cross the FishAudio post-module sliding window (128),
    // exercising bounded-context streaming decode logic.
    let frame_cap = 192usize;
    let non_streaming_config = TtsRunConfig {
        streaming_enabled: false,
        max_semantic_frames: frame_cap,
        ..TtsRunConfig::default()
    };
    let streaming_config = TtsRunConfig {
        streaming_enabled: true,
        initial_chunk_frames: 1,
        min_chunk_frames: 16,
        max_chunk_frames: 16,
        max_semantic_frames: frame_cap,
        ..TtsRunConfig::default()
    };

    let baseline = session
        .synthesize_with_seed_and_config(Input::Text(fixture.text.clone()), fixture.seed, &non_streaming_config)
        .expect("baseline");
    let mut streamed_samples = Vec::<f32>::new();
    let streamed = session
        .synthesize_streaming_with_seed_and_config(
            Input::Text(fixture.text.clone()),
            fixture.seed,
            &streaming_config,
            |chunk| {
                streamed_samples.extend_from_slice(chunk.samples());
            },
        )
        .expect("streamed");

    assert_eq!(streamed.sample_rate(), baseline.sample_rate());
    assert_eq!(streamed.channels(), baseline.channels());
    assert_eq!(streamed.lengths(), baseline.lengths());
    assert_eq!(streamed_samples.len(), streamed.samples().len());
    assert_eq!(streamed_samples, streamed.samples());

    let max_abs_diff = streamed
        .samples()
        .iter()
        .zip(baseline.samples().iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_abs_diff <= 1e-4, "FishAudio f32 streaming/non-streaming mismatch: max_abs_diff={max_abs_diff}");
}

#[test]
fn tts_session_fishaudio_f16_greedy_emits_bounded_semantic_frames() {
    let _guard = TEST_MUTEX.lock().expect("lock");
    let Some(model_path) = load_optional_fishaudio_model_path_f16() else {
        println!("Skipping FishAudio f16 semantic-frame test: set LALAMO_UZU_FISHAUDIO_MODEL_PATH_F16");
        return;
    };

    let session = TtsSession::new_with_options(model_path, greedy_fishaudio_session_options()).expect("tts session");
    let tokens = session
        .generate_semantic_tokens_with_seed(Input::Text("I will tell you about London, get ready!".to_string()), 123)
        .expect("generate semantic tokens");

    assert!(tokens.frames() >= 8, "expected at least 8 semantic frames, got {}", tokens.frames());
    assert!(tokens.frames() <= 128, "unexpectedly large greedy semantic frame count: {}", tokens.frames());
}

#[test]
fn tts_session_fishaudio_f16_streaming_matches_non_streaming_audio() {
    let _guard = TEST_MUTEX.lock().expect("lock");
    let Some(model_path) = load_optional_fishaudio_model_path_f16() else {
        println!("Skipping FishAudio f16 streaming parity test: set LALAMO_UZU_FISHAUDIO_MODEL_PATH_F16");
        return;
    };

    let session = TtsSession::new_with_options(model_path, greedy_fishaudio_session_options()).expect("tts session");
    let text = Input::Text("I will tell you about London, get ready!".to_string());
    let seed = 123_u64;

    let non_streaming_config = TtsRunConfig {
        streaming_enabled: false,
        max_semantic_frames: 64,
        ..TtsRunConfig::default()
    };
    let streaming_config = TtsRunConfig {
        streaming_enabled: true,
        initial_chunk_frames: 1,
        min_chunk_frames: 16,
        max_chunk_frames: 16,
        max_semantic_frames: 64,
        ..TtsRunConfig::default()
    };

    let baseline =
        session.synthesize_with_seed_and_config(text.clone(), seed, &non_streaming_config).expect("baseline");
    let mut streamed_samples = Vec::<f32>::new();
    let streamed = session
        .synthesize_streaming_with_seed_and_config(text, seed, &streaming_config, |chunk| {
            streamed_samples.extend_from_slice(chunk.samples());
        })
        .expect("streamed");

    assert_eq!(streamed.sample_rate(), baseline.sample_rate());
    assert_eq!(streamed.channels(), baseline.channels());
    assert_eq!(streamed.lengths(), baseline.lengths());
    assert_eq!(streamed_samples, streamed.samples());

    let max_abs_diff = streamed
        .samples()
        .iter()
        .zip(baseline.samples().iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_abs_diff <= 2e-4, "FishAudio f16 streaming/non-streaming mismatch: max_abs_diff={max_abs_diff}");
}

#[test]
fn tts_session_fishaudio_f16_async_followups_match_sequential_with_repetition_penalty() {
    let _guard = TEST_MUTEX.lock().expect("lock");
    let Some(model_path) = load_optional_fishaudio_model_path_f16() else {
        println!("Skipping FishAudio f16 followup parity test: set LALAMO_UZU_FISHAUDIO_MODEL_PATH_F16");
        return;
    };

    let text = Input::Text("Fish Audio works and it matches the reference implementation.".to_string());
    let seed = 123_u64;
    let config = TtsRunConfig {
        max_semantic_frames: 64,
        ..TtsRunConfig::default()
    };

    let mut sequential_options = TtsSessionOptions::default();
    sequential_options.text_decoder.followup_strategy = TextDecoderFollowupStrategy::SequentialExact;
    let sequential_session =
        TtsSession::new_with_options(model_path.clone(), sequential_options).expect("sequential session");
    let sequential_tokens = sequential_session
        .generate_semantic_tokens_with_seed_and_config(text.clone(), seed, &config)
        .expect("sequential tokens");

    let mut async_options = TtsSessionOptions::default();
    async_options.text_decoder.followup_strategy = TextDecoderFollowupStrategy::AsyncChain;
    let async_session = TtsSession::new_with_options(model_path, async_options).expect("async session");
    let async_tokens =
        async_session.generate_semantic_tokens_with_seed_and_config(text, seed, &config).expect("async tokens");

    assert_eq!(async_tokens.batch_size(), sequential_tokens.batch_size());
    assert_eq!(async_tokens.codebooks(), sequential_tokens.codebooks());
    assert_eq!(async_tokens.frames(), sequential_tokens.frames());
    assert_eq!(async_tokens.lengths(), sequential_tokens.lengths());
    assert_eq!(async_tokens.tokens(), sequential_tokens.tokens());
}

#[test]
fn tts_session_fishaudio_bf16_greedy_emits_bounded_semantic_frames() {
    let _guard = TEST_MUTEX.lock().expect("lock");
    let Some(model_path) = load_optional_fishaudio_model_path_bf16() else {
        println!("Skipping FishAudio bf16 semantic-frame test: set LALAMO_UZU_FISHAUDIO_MODEL_PATH_BF16");
        return;
    };

    let session = TtsSession::new_with_options(model_path, greedy_fishaudio_session_options()).expect("tts session");
    let tokens = session
        .generate_semantic_tokens_with_seed(Input::Text("I will tell you about London, get ready!".to_string()), 123)
        .expect("generate semantic tokens");

    assert!(tokens.frames() >= 8, "expected at least 8 semantic frames, got {}", tokens.frames());
    assert!(tokens.frames() <= 128, "unexpectedly large greedy semantic frame count: {}", tokens.frames());
}

#[test]
fn tts_session_fishaudio_bf16_streaming_matches_non_streaming_audio() {
    let _guard = TEST_MUTEX.lock().expect("lock");
    let Some(model_path) = load_optional_fishaudio_model_path_bf16() else {
        println!("Skipping FishAudio bf16 streaming parity test: set LALAMO_UZU_FISHAUDIO_MODEL_PATH_BF16");
        return;
    };

    let session = TtsSession::new_with_options(model_path, greedy_fishaudio_session_options()).expect("tts session");
    let text = Input::Text("I will tell you about London, get ready!".to_string());
    let seed = 123_u64;

    let non_streaming_config = TtsRunConfig {
        streaming_enabled: false,
        max_semantic_frames: 64,
        ..TtsRunConfig::default()
    };
    let streaming_config = TtsRunConfig {
        streaming_enabled: true,
        initial_chunk_frames: 1,
        min_chunk_frames: 16,
        max_chunk_frames: 16,
        max_semantic_frames: 64,
        ..TtsRunConfig::default()
    };

    let baseline =
        session.synthesize_with_seed_and_config(text.clone(), seed, &non_streaming_config).expect("baseline");
    let mut streamed_samples = Vec::<f32>::new();
    let streamed = session
        .synthesize_streaming_with_seed_and_config(text, seed, &streaming_config, |chunk| {
            streamed_samples.extend_from_slice(chunk.samples());
        })
        .expect("streamed");

    assert_eq!(streamed.sample_rate(), baseline.sample_rate());
    assert_eq!(streamed.channels(), baseline.channels());
    assert_eq!(streamed.lengths(), baseline.lengths());
    assert_eq!(streamed_samples, streamed.samples());

    let max_abs_diff = streamed
        .samples()
        .iter()
        .zip(baseline.samples().iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_abs_diff <= 2e-3, "FishAudio bf16 streaming/non-streaming mismatch: max_abs_diff={max_abs_diff}");
}
