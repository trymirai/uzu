#![cfg(metal_backend)]

use std::sync::Arc;

use proc_macros::uzu_test;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use test_tag::tag;

use crate::{
    common::{path::get_test_model_path, repeat_speculator::RepeatSpeculator},
    session::{
        Session,
        config::{DecodingConfig, GrammarConfig, RunConfig, SpeculatorConfig, StructuredOutput},
        parameter::{SamplingPolicy, SamplingSeed},
        types::Input,
    },
};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Address {
    street: String,
    city: String,
    country: String,
    postal_code: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Person {
    name: String,
    age: u32,
    email: String,
    address: Address,
    hobbies: Vec<String>,
}

fn test_grammar(speculator_config: SpeculatorConfig) {
    let model_dir = get_test_model_path();

    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model not found at {}.\nRun scripts/download_test_model.sh first.",
            model_dir.display()
        );
        return;
    }

    let decoding_config = DecodingConfig::default()
        .with_sampling_seed(SamplingSeed::Custom(42))
        .with_speculator_config(speculator_config);
    let mut session = Session::new(model_dir, decoding_config).expect("Failed to create session");

    let input = Input::Text("Generate a detailed person profile with address and hobbies in JSON format.".to_string());

    let grammar_config = GrammarConfig::from_json_schema_type::<Person>().expect("Failed to create grammar config");

    let run_config =
        RunConfig::default().tokens_limit(1024).sampling_policy(SamplingPolicy::Default).grammar_config(grammar_config);

    let output = session
        .run(input, run_config, None::<fn(backend_uzu::session::types::Output) -> bool>)
        .expect("Failed to run session");

    let stats = output.stats;
    let total_tokens = stats.total_stats.tokens_count_input + stats.total_stats.tokens_count_output;

    println!("\n=== Test Results ===");
    println!("Generated output length: {} chars", output.text.original.len());
    println!(
        "Input tokens: {}, Output tokens: {}",
        stats.total_stats.tokens_count_input, stats.total_stats.tokens_count_output
    );
    println!("Total time: {:.2}s", stats.total_stats.duration);
    println!("Throughput: {:.2} tokens/s", total_tokens as f64 / stats.total_stats.duration);

    match serde_json::from_str::<Person>(&output.text.original) {
        Ok(person) => {
            println!("✓ Valid Person JSON!");
            println!("  Name: {}", person.name);
            println!("  Age: {}", person.age);
            println!("  Email: {}", person.email);
            println!("  Address: {}, {}", person.address.city, person.address.country);
            println!("  Hobbies: {:?}", person.hobbies);

            assert!(!person.name.is_empty());
            assert!(person.age > 0);
            assert!(person.email.contains('@'));
            assert!(!person.address.city.is_empty());
            assert!(!person.hobbies.is_empty());
        },
        Err(e) => {
            panic!("✗ Generated invalid JSON: {}\nOutput: {}", e, output.text.original);
        },
    }
}

#[tag(heavy)]
#[uzu_test]
fn test_grammar_json_schema() {
    test_grammar(SpeculatorConfig::default());
}

#[tag(heavy)]
#[uzu_test]
fn test_grammar_json_schema_with_speculator() {
    test_grammar(SpeculatorConfig {
        number_of_speculated_tokens: 16,
        speculator: Arc::new(RepeatSpeculator),
    });
}

fn unwrapped_schema(raw: &str) -> String {
    match GrammarConfig::structured_output_from_schema(raw) {
        StructuredOutput::Schema(schema) => schema,
        StructuredOutput::AnyJson => panic!("expected schema for: {raw}"),
    }
}

#[uzu_test]
fn unwraps_nested_response_format_envelope() {
    let raw = r#"{"type":"json_schema","json_schema":{"name":"person","strict":true,"schema":{"type":"object","properties":{"name":{"type":"string"}}}}}"#;
    let normalized: serde_json::Value = serde_json::from_str(&unwrapped_schema(raw)).unwrap();
    assert_eq!(normalized, serde_json::json!({"type":"object","properties":{"name":{"type":"string"}}}));
}

#[uzu_test]
fn unwraps_flattened_response_format_envelope() {
    let raw = r#"{"type":"json_schema","name":"person","schema":{"type":"object"}}"#;
    let normalized: serde_json::Value = serde_json::from_str(&unwrapped_schema(raw)).unwrap();
    assert_eq!(normalized, serde_json::json!({"type":"object"}));
}

#[uzu_test]
fn unwraps_bare_inner_json_schema_object() {
    let raw = r#"{"name":"person","schema":{"type":"object"}}"#;
    let normalized: serde_json::Value = serde_json::from_str(&unwrapped_schema(raw)).unwrap();
    assert_eq!(normalized, serde_json::json!({"type":"object"}));
}

#[uzu_test]
fn json_object_maps_to_any_json() {
    assert_eq!(GrammarConfig::structured_output_from_schema(r#"{"type":"json_object"}"#), StructuredOutput::AnyJson,);
}

#[uzu_test]
fn passes_through_bare_json_schema() {
    let raw = r#"{"type":"object","properties":{"schema":{"type":"string"}}}"#;
    let normalized: serde_json::Value = serde_json::from_str(&unwrapped_schema(raw)).unwrap();
    assert_eq!(normalized, serde_json::from_str::<serde_json::Value>(raw).unwrap());
}

#[uzu_test]
fn passes_through_unparseable_input() {
    assert_eq!(unwrapped_schema("not json"), "not json");
}
