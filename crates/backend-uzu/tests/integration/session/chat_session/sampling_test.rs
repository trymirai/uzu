#![cfg(metal_backend)]

use std::path::{Path, PathBuf};

use backend_uzu::session::{
    Session,
    config::{DecodingConfig, RunConfig},
    parameter::{SamplingMethod, SamplingPolicy, SamplingProcessingOrder, SamplingSeed},
    types::{Input, Message, Output},
};
use proc_macros::uzu_test;
use test_tag::tag;

fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("THINKING_TEST_MODEL").ok()?);
    path.join("config.json").exists().then_some(path)
}

fn run(
    path: &Path,
    prompt: &str,
    seed: u64,
    policy: SamplingPolicy,
) -> String {
    let mut session =
        Session::new(path.to_path_buf(), DecodingConfig::default().with_sampling_seed(SamplingSeed::Custom(seed)))
            .expect("load model");
    session
        .run(
            Input::Messages(vec![Message::user(prompt.to_string())]),
            RunConfig::default().tokens_limit(48).enable_thinking(false).sampling_policy(policy),
            Some(|_: Output| true),
        )
        .expect("run")
        .text
        .original
}

fn greedy() -> SamplingPolicy {
    SamplingPolicy::Custom {
        value: SamplingMethod::Greedy,
    }
}

fn stochastic(temperature: f32) -> SamplingPolicy {
    SamplingPolicy::Custom {
        value: SamplingMethod::Stochastic {
            temperature: Some(temperature),
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: None,       // TODO
            suffix_repetition_length: None, // TODO
            processing_order: SamplingProcessingOrder::TemperatureThenFilters,
        },
    }
}

const PROMPT: &str = "Write one creative sentence about the ocean.";

#[ignore = "requires THINKING_TEST_MODEL pointing at a downloaded model"]
#[tag(heavy)]
#[uzu_test]
fn greedy_sampling_is_deterministic() {
    let Some(path) = model_path() else {
        panic!("set THINKING_TEST_MODEL to a downloaded model directory");
    };
    let a = run(&path, PROMPT, 1, greedy());
    let b = run(&path, PROMPT, 2, greedy());
    println!("greedy (seed 1): {a}\ngreedy (seed 2): {b}");
    assert_eq!(a, b, "greedy sampling must be deterministic regardless of seed");
}

#[ignore = "requires THINKING_TEST_MODEL pointing at a downloaded model"]
#[tag(heavy)]
#[uzu_test]
fn stochastic_sampling_varies_with_seed() {
    let Some(path) = model_path() else {
        panic!("set THINKING_TEST_MODEL to a downloaded model directory");
    };
    let a = run(&path, PROMPT, 1, stochastic(1.0));
    let b = run(&path, PROMPT, 2, stochastic(1.0));
    let g = run(&path, PROMPT, 1, greedy());
    println!("stochastic (seed 1): {a}\nstochastic (seed 2): {b}\ngreedy:             {g}");
    assert_ne!(a, b, "temperature sampling must vary across seeds");
}
