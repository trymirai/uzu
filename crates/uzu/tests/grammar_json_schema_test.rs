#![cfg(target_os = "macos")]

use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uzu::session::{
    Session,
    config::{DecodingConfig, GrammarConfig, RunConfig},
    parameter::SamplingPolicy,
    types::Input,
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

#[test]
#[ignore = "Test requires model; run manually with RUST_TEST_THREADS=1"]
fn test_grammar_json_schema() {
    let crate_version = env!("CARGO_PKG_VERSION");
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.parent().unwrap().parent().unwrap();
    let model_dir = repo_root
        .join("models")
        .join(crate_version)
        .join("Llama-3.2-1B-Instruct");

    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model not found at {}.\nRun scripts/download_test_model.sh first.",
            model_dir.display()
        );
        return;
    }

    let decoding_config = DecodingConfig::default();
    let mut session = Session::new(model_dir, decoding_config)
        .expect("Failed to create session");

    let input = Input::Text(
        "Generate a detailed person profile with address and hobbies in JSON format.".to_string(),
    );

    let grammar_config_simple =
        GrammarConfig::from_json_schema_type::<Person>()
            .expect("Failed to create grammar config");

    let _grammar_config_advanced =
        GrammarConfig::from_json_schema_type_with_config::<Person>(
            true,
            Some(4),
            Some((",".to_string(), ": ".to_string())),
            true,
        )
        .expect("Failed to create grammar config");

    let run_config = RunConfig::default()
        .tokens_limit(1000)
        .sampling_policy(SamplingPolicy::Default)
        .grammar_config(grammar_config_simple);

    let output = session
        .run(input, run_config, None::<fn(uzu::session::types::Output) -> bool>)
        .expect("Failed to run session");

    let stats = output.stats;
    let total_tokens = stats.total_stats.tokens_count_input
        + stats.total_stats.tokens_count_output;

    println!("\n=== Test Results ===");
    println!("Generated output length: {} chars", output.text.original.len());
    println!(
        "Input tokens: {}, Output tokens: {}",
        stats.total_stats.tokens_count_input,
        stats.total_stats.tokens_count_output
    );
    println!("Total time: {:.2}s", stats.total_stats.duration);
    println!(
        "Throughput: {:.2} tokens/s",
        total_tokens as f64 / stats.total_stats.duration
    );

    match serde_json::from_str::<Person>(&output.text.original) {
        Ok(person) => {
            println!("✓ Valid Person JSON!");
            println!("  Name: {}", person.name);
            println!("  Age: {}", person.age);
            println!("  Email: {}", person.email);
            println!(
                "  Address: {}, {}",
                person.address.city, person.address.country
            );
            println!("  Hobbies: {:?}", person.hobbies);

            assert!(!person.name.is_empty());
            assert!(person.age > 0);
            assert!(person.email.contains('@'));
            assert!(!person.address.city.is_empty());
            assert!(!person.hobbies.is_empty());
        },
        Err(e) => {
            panic!(
                "✗ Generated invalid JSON: {}\nOutput: {}",
                e, output.text.original
            );
        },
    }
}
