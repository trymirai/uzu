mod common;
use std::path::PathBuf;

use uzu::session::{
    Session,
    config::{DecodingConfig, RunConfig, SpeculatorConfig},
    parameter::{
        ContextLength, PrefillStepSize, SamplingMethod, SamplingPolicy,
        SamplingSeed,
    },
    types::{Input, Message, Output},
};

fn build_model_path() -> PathBuf {
    common::get_test_model_path()
}

fn build_decoding_config() -> DecodingConfig {
    DecodingConfig::new(
        PrefillStepSize::default(),
        ContextLength::default(),
        SpeculatorConfig::default(),
        SamplingSeed::Custom(42),
        true,
    )
}

#[test]
fn test_text_session_base() {
    let text = String::from("Tell about London");
    run(text, build_decoding_config(), 128);
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

fn run(
    text: String,
    decoding_config: DecodingConfig,
    tokens_limit: u64,
) {
    let mut session =
        Session::new(build_model_path(), decoding_config).unwrap();
    let input = Input::Text(text);
    let output = session
        .run(
            input,
            RunConfig::new(
                tokens_limit,
                true,
                SamplingPolicy::Custom {
                    value: SamplingMethod::Greedy,
                },
            ),
            Some(|_: Output| {
                return true;
            }),
        )
        .unwrap();

    println!("-------------------------");
    println!("{:#?}", output.chain_of_thought);
    println!("-------------------------");
    println!("{:#?}", output.response);
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
        Session::new(build_model_path(), build_decoding_config()).unwrap();

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
            output.response.clone().unwrap_or(String::new()),
            output.chain_of_thought.clone(),
        ));
        println!("Assistant > {}", output.response.clone().unwrap());
    }
}
