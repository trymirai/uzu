use std::{fs, path::PathBuf};

use proc_macros::uzu_test;

use crate::session::{
    Session,
    config::{DecodingConfig, RunConfig},
    types::{Input, Message, Output},
};

const PROMPT: &str = "Write a detailed essay about the history and future of computing.";
const DECODE_TOKEN_LIMIT: u64 = 512;

#[uzu_test]
#[ignore = "whole-model GPU profiling; run via `cargo profile-model`"]
fn profile_whole_model() {
    let mut session = Session::new(model_path(), DecodingConfig::default()).unwrap();

    let output = session
        .run(
            Input::Messages(vec![Message::user(PROMPT.to_string())]),
            RunConfig::default().tokens_limit(DECODE_TOKEN_LIMIT),
            Some(|_: Output| true),
        )
        .unwrap();

    let statistics = output.stats;
    let prefill_tokens = statistics.total_stats.tokens_count_input;
    let decode_tokens = statistics.total_stats.tokens_count_output;
    let prefill_seconds = statistics.prefill_stats.duration;
    let decode_seconds = statistics.generate_stats.as_ref().map(|stats| stats.duration).unwrap_or(0.0);
    let total_seconds = statistics.total_stats.duration;
    let total_joules = statistics.power_stats.as_ref().map(|stats| stats.energy_joules).unwrap_or(0.0);
    let prefill_joules = statistics.power_stats.as_ref().and_then(|stats| stats.prefill_energy_joules).unwrap_or(0.0);
    let decode_joules = statistics.power_stats.as_ref().and_then(|stats| stats.decode_energy_joules).unwrap_or(0.0);
    let decode_joules_per_token = if decode_tokens == 0 {
        0.0
    } else {
        decode_joules / decode_tokens as f64
    };

    eprintln!(
        "prefill_tokens={prefill_tokens} decode_tokens={decode_tokens} \
         prefill_seconds={prefill_seconds:.4} decode_seconds={decode_seconds:.4} total_seconds={total_seconds:.4} \
         prefill_joules={prefill_joules:.4} decode_joules={decode_joules:.4} total_joules={total_joules:.4}"
    );

    let lines = vec![
        "prefill_tokens,decode_tokens,prefill_seconds,decode_seconds,total_seconds,prefill_joules,decode_joules,total_joules,decode_joules_per_token".to_string(),
        format!(
            "{prefill_tokens},{decode_tokens},{prefill_seconds},{decode_seconds},{total_seconds},{prefill_joules},{decode_joules},{total_joules},{decode_joules_per_token}"
        ),
    ];

    let output_path = output_directory().join("whole_model.csv");
    fs::write(&output_path, lines.join("\n")).unwrap();
    eprintln!("wrote {}", output_path.display());
}

fn model_path() -> PathBuf {
    let path = std::env::var_os("UZU_PROFILE_MODEL")
        .map(PathBuf::from)
        .expect("set UZU_PROFILE_MODEL to a uzu model directory");
    assert!(path.join("config.json").exists(), "UZU_PROFILE_MODEL missing config.json: {}", path.display());
    path
}

fn output_directory() -> PathBuf {
    std::env::var_os("UZU_PROFILE_OUTPUT_DIRECTORY").map(PathBuf::from).unwrap_or_else(std::env::temp_dir)
}
