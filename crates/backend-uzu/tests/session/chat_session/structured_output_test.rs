#![cfg(all(metal_backend, grammar_xgrammar))]

use std::path::{Path, PathBuf};

use backend_uzu::session::{
    Session,
    config::{DecodingConfig, GrammarConfig, RunConfig},
    parameter::SamplingSeed,
    types::{Input, Message, Output},
};
use proc_macros::uzu_test;
use serde::{Deserialize, Serialize};
use test_tag::tag;

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct Country {
    name: String,
    capital: String,
}

fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("THINKING_TEST_MODEL").ok()?);
    path.join("config.json").exists().then_some(path)
}

fn ask_json(
    path: &Path,
    enable_thinking: bool,
) -> Output {
    let schema = GrammarConfig::from_json_schema_type::<Country>().expect("build schema grammar");
    let mut session =
        Session::new(path.to_path_buf(), DecodingConfig::default().with_sampling_seed(SamplingSeed::Custom(42)))
            .expect("load model");
    session
        .run(
            Input::Messages(vec![Message::user(
                "Give me a JSON object for France with name and capital fields.".to_string(),
            )]),
            RunConfig::default().tokens_limit(512).enable_thinking(enable_thinking).grammar_config(schema),
            Some(|_: Output| true),
        )
        .expect("run")
}

#[ignore = "requires THINKING_TEST_MODEL pointing at a downloaded thinking model"]
#[tag(heavy)]
#[uzu_test]
fn structured_output_with_reasoning_enabled_yields_valid_json() {
    let Some(path) = model_path() else {
        panic!("set THINKING_TEST_MODEL to a downloaded thinking-model directory");
    };

    let output = ask_json(&path, true);
    println!("[raw]\n{}", output.text.original);
    println!("[chain_of_thought] {:?}", output.text.parsed.chain_of_thought);
    println!("[response] {:?}", output.text.parsed.response);

    let response = output.text.parsed.response.as_deref().expect("response present");
    serde_json::from_str::<Country>(response).expect("response must be schema-valid JSON");
}

#[ignore = "requires THINKING_TEST_MODEL pointing at a downloaded thinking model"]
#[tag(heavy)]
#[uzu_test]
fn structured_output_with_reasoning_disabled_yields_bare_json() {
    let Some(path) = model_path() else {
        panic!("set THINKING_TEST_MODEL to a downloaded thinking-model directory");
    };

    let output = ask_json(&path, false);
    println!("[raw]\n{}", output.text.original);
    println!("[response] {:?}", output.text.parsed.response);

    assert!(
        output.text.parsed.chain_of_thought.is_none(),
        "disabled-thinking output must not render a reasoning chain, got {:?}",
        output.text.parsed.chain_of_thought,
    );
    let response = output.text.parsed.response.as_deref().expect("response present");
    serde_json::from_str::<Country>(response).expect("response must be schema-valid JSON");
}
