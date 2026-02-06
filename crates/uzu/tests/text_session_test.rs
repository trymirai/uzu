mod common;
use std::path::PathBuf;

use uzu::session::{
    Session,
    config::{DecodingConfig, RunConfig},
    parameter::SamplingSeed,
    types::{Input, Message, Output},
};

fn build_model_path() -> PathBuf {
    common::get_test_model_path()
}

fn build_decoding_config() -> DecodingConfig {
    DecodingConfig::default().with_sampling_seed(SamplingSeed::Custom(42))
}

fn build_default_text() -> String {
    let text = String::from("Tell about London");
    return text;
}

#[test]
fn test_text_session_base() {
    run(build_default_text(), build_decoding_config(), 128);
}

#[test]
fn test_text_session_scenario() {
    let system_prompt = String::from("You are a helpful assistant.");
    let user_prompts = vec![
        String::from("Tell about London"),
        String::from("Compare with New York"),
    ];
    run_scenario(Some(system_prompt), user_prompts);
}

#[test]
fn test_text_session_stability() {
    let mut session =
        Session::new(build_model_path(), build_decoding_config(), None)
            .unwrap();
    println!("Index | TTFT, s | Prompt t/s | Generate t/s");
    for index in 0..10 {
        let input = Input::Text(build_default_text());
        let output = session
            .run(
                input,
                RunConfig::default().tokens_limit(128),
                Some(|_: Output| {
                    return true;
                }),
            )
            .unwrap();
        println!(
            "{:.5} | {:.5} | {:.5} | {:.5}",
            index,
            output.stats.prefill_stats.duration,
            output.stats.prefill_stats.processed_tokens_per_second,
            output.stats.generate_stats.unwrap().tokens_per_second
        );
    }
}

fn run(
    text: String,
    decoding_config: DecodingConfig,
    tokens_limit: u64,
) {
    let mut session =
        Session::new(build_model_path(), decoding_config, None).unwrap();
    let input = Input::Text(text);
    let output = session
        .run(
            input,
            RunConfig::default().tokens_limit(tokens_limit),
            Some(|_: Output| {
                return true;
            }),
        )
        .unwrap();

    let empty_response = String::from("None");

    println!("-------------------------");
    println!(
        "{}",
        output.text.parsed.chain_of_thought().unwrap_or(empty_response.clone())
    );
    println!("-------------------------");
    println!(
        "{}",
        output.text.parsed.response().unwrap_or(empty_response.clone())
    );
    println!("-------------------------");
    println!("{:#?}", output.stats);
    println!("-------------------------");
    println!("Finish reason: {:?}", output.finish_reason);
    println!("-------------------------");
}

fn run_scenario(
    system_prompt: Option<String>,
    user_prompts: Vec<String>,
) {
    let mut session =
        Session::new(build_model_path(), build_decoding_config(), None)
            .unwrap();

    let mut messages: Vec<Message> = vec![];
    if let Some(system_prompt) = system_prompt {
        messages.push(Message::system(system_prompt.clone()));
        println!("System > {}", system_prompt.clone());
    }

    for user_prompt in user_prompts {
        messages.push(Message::user(user_prompt.clone()));
        println!("User > {}", user_prompt.clone());

        let input = Input::Messages(messages.clone());
        let output = session
            .run(
                input,
                RunConfig::default(),
                Some(|_: Output| {
                    return true;
                }),
            )
            .unwrap();
        messages.push(Message::assistant(
            output.text.parsed.response().unwrap_or(String::new()),
            output.text.parsed.chain_of_thought(),
        ));
        println!("Assistant > {}", output.text.original.clone());
    }
}
