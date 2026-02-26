#![cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]

use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};
use uzu::audio::{
    AudioCodecRuntime, AudioTokenGrid, AudioTokenPacking, NanoCodecFsqRuntime, nanocodec::fsq::fsq_decode_reference,
};
use uzu::session::TtsCodecSession;

#[derive(Debug, Deserialize)]
struct Fixture {
    tts_config: serde_json::Value,
    tokens: TokenFixture,
    expected: ExpectedFixture,
}

#[derive(Debug, Deserialize)]
struct TokenFixture {
    batch_size: usize,
    codebooks: usize,
    frames: usize,
    lengths: Vec<usize>,
    packing: String,
    tokens: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct ExpectedFixture {
    latent_nct: TensorFixture,
    pcm: PcmFixture,
}

#[derive(Debug, Deserialize)]
struct TensorFixture {
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct PcmFixture {
    sample_rate: u32,
    channels: usize,
    lengths: Vec<usize>,
    samples: Vec<f32>,
}

fn load_fixture_from_path(path: &Path) -> Fixture {
    let fixture_str =
        fs::read_to_string(path).unwrap_or_else(|err| panic!("failed to read fixture from {}: {err}", path.display()));
    serde_json::from_str(&fixture_str).expect("parse lalamo fixture")
}

fn load_fixture() -> Fixture {
    if let Ok(path) = std::env::var("LALAMO_UZU_FIXTURE_PATH") {
        return load_fixture_from_path(Path::new(&path));
    }

    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lalamo_nanocodec_export_fixture.json");
    load_fixture_from_path(&path)
}

fn load_optional_lalamo_model_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("LALAMO_UZU_MODEL_PATH") {
        let path = PathBuf::from(path);
        return if path.join("config.json").exists() && path.join("model.safetensors").exists() {
            Some(path)
        } else {
            None
        };
    }

    let default = PathBuf::from("/tmp/lalamo_nanocodec_convert");
    if default.join("config.json").exists() && default.join("model.safetensors").exists() {
        Some(default)
    } else {
        None
    }
}

fn parse_packing(value: &str) -> AudioTokenPacking {
    match value {
        "frame_major" => AudioTokenPacking::FrameMajor,
        "codebook_major" => AudioTokenPacking::CodebookMajor,
        other => panic!("unsupported token packing in fixture: {other}"),
    }
}

fn assert_close(
    got: &[f32],
    expected: &[f32],
    atol: f32,
) {
    assert_eq!(got.len(), expected.len(), "length mismatch");
    for (index, (&lhs, &rhs)) in got.iter().zip(expected.iter()).enumerate() {
        let delta = (lhs - rhs).abs();
        assert!(
            delta <= atol,
            "value mismatch at index={index}: got={lhs}, expected={rhs}, delta={delta}, atol={atol}"
        );
    }
}

#[test]
fn lalamo_fixture_fsq_decode_matches_latent_trace() {
    let fixture = load_fixture();
    let runtime = NanoCodecFsqRuntime::from_tts_config_value(&fixture.tts_config).expect("runtime");

    assert_eq!(
        fixture.expected.latent_nct.shape,
        vec![fixture.tokens.batch_size, runtime.config().channels(), fixture.tokens.frames]
    );
    assert_eq!(
        fixture.expected.latent_nct.values.len(),
        fixture.tokens.batch_size * runtime.config().channels() * fixture.tokens.frames
    );

    let tokens_i32: Vec<i32> = fixture.tokens.tokens.iter().map(|&value| value as i32).collect();
    let lengths_i32: Vec<i32> = fixture.tokens.lengths.iter().map(|&value| value as i32).collect();
    let latent = fsq_decode_reference(
        &tokens_i32,
        &lengths_i32,
        fixture.tokens.batch_size,
        fixture.tokens.codebooks,
        fixture.tokens.frames,
        runtime.config().codebook_dim_per_group(),
        runtime.config().num_levels_per_group(),
    )
    .expect("fsq decode");

    assert_close(&latent, &fixture.expected.latent_nct.values, 1e-6);
}

#[test]
fn lalamo_fixture_runtime_decode_matches_expected_pcm() {
    let fixture = load_fixture();
    let runtime = NanoCodecFsqRuntime::from_tts_config_value(&fixture.tts_config).expect("runtime");

    let tokens = AudioTokenGrid::new(
        fixture.tokens.tokens.clone().into_boxed_slice(),
        fixture.tokens.batch_size,
        fixture.tokens.codebooks,
        fixture.tokens.frames,
        fixture.tokens.lengths.clone().into_boxed_slice(),
        parse_packing(&fixture.tokens.packing),
    )
    .expect("token grid");

    let decoded = runtime.decode(&tokens).expect("decode");

    assert_eq!(decoded.sample_rate(), fixture.expected.pcm.sample_rate);
    assert_eq!(decoded.channels(), fixture.expected.pcm.channels);
    assert_eq!(decoded.lengths(), fixture.expected.pcm.lengths.as_slice());
    assert_close(decoded.samples(), &fixture.expected.pcm.samples, 2e-4);
}

#[test]
fn lalamo_normal_export_decode_matches_expected_pcm() {
    let Some(model_path) = load_optional_lalamo_model_path() else {
        println!(
            "Skipping normal-export parity test: set LALAMO_UZU_MODEL_PATH (or create /tmp/lalamo_nanocodec_convert)"
        );
        return;
    };
    let fixture_path = match std::env::var("LALAMO_UZU_FIXTURE_PATH") {
        Ok(path) => PathBuf::from(path),
        Err(_) => {
            println!(
                "Skipping normal-export parity test: set LALAMO_UZU_FIXTURE_PATH to a Lalamo fixture for the same model"
            );
            return;
        },
    };

    let fixture = load_fixture_from_path(&fixture_path);
    let session = TtsCodecSession::new(model_path.clone()).unwrap_or_else(|err| {
        panic!("failed to create TtsCodecSession from normal export {}: {err:?}", model_path.display())
    });

    assert!(
        session.runtime().config().decoder().is_some(),
        "normal export runtime should load decoder weights from model.safetensors"
    );

    let tokens = AudioTokenGrid::new(
        fixture.tokens.tokens.clone().into_boxed_slice(),
        fixture.tokens.batch_size,
        fixture.tokens.codebooks,
        fixture.tokens.frames,
        fixture.tokens.lengths.clone().into_boxed_slice(),
        parse_packing(&fixture.tokens.packing),
    )
    .expect("token grid");

    let decoded = session.decode(&tokens).expect("decode");

    assert_eq!(decoded.sample_rate(), fixture.expected.pcm.sample_rate);
    assert_eq!(decoded.channels(), fixture.expected.pcm.channels);
    assert_eq!(decoded.lengths(), fixture.expected.pcm.lengths.as_slice());
    assert_close(decoded.samples(), &fixture.expected.pcm.samples, 2e-4);
}
