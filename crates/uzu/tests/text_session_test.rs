mod common;
use std::path::PathBuf;

use uzu::{
    backends::metal::sampling_config::SamplingConfig,
    generator::config::{ContextLength, SamplingSeed, SpeculatorConfig},
    session::{
        session::Session,
        session_config::SessionConfig,
        session_input::SessionInput,
        session_message::{SessionMessage, SessionMessageRole},
        session_output::SessionOutput,
        session_run_config::SessionRunConfig,
    },
};

fn build_model_path() -> PathBuf {
    common::get_test_model_path()
}

fn build_session_config() -> SessionConfig {
    SessionConfig::new(
        64,
        SpeculatorConfig::default(),
        true,
        SamplingSeed::Custom(42),
        ContextLength::Default,
    )
}

#[test]
fn test_text_session_base() {
    let text = String::from("Tell about London");
    run(text, build_session_config(), 128);
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
    config: SessionConfig,
    tokens_limit: u64,
) {
    let mut session = Session::new(build_model_path()).unwrap();
    session.load_with_session_config(config).unwrap();

    let input = SessionInput::Text(text);
    let output = session
        .run(
            input,
            SessionRunConfig::new(
                tokens_limit,
                true,
                Some(SamplingConfig::Argmax),
            ),
            Some(|_: SessionOutput| {
                return true;
            }),
        )
        .unwrap();

    println!("-------------------------");
    println!("{}", output.text);
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
    let config = build_session_config();
    let mut session = Session::new(build_model_path()).unwrap();
    session.load_with_session_config(config).unwrap();

    let mut messages: Vec<SessionMessage> = vec![];
    if let Some(system_prompt) = system_prompt {
        messages.push(SessionMessage {
            role: SessionMessageRole::System,
            content: system_prompt.clone(),
        });
        println!("System > {}", system_prompt.clone());
    }

    for user_prompt in user_prompts {
        messages.push(SessionMessage {
            role: SessionMessageRole::User,
            content: user_prompt.clone(),
        });
        println!("User > {}", user_prompt.clone());

        let input = SessionInput::Messages(messages.clone());
        let output = session
            .run(
                input,
                SessionRunConfig::default(),
                Some(|_: SessionOutput| {
                    return true;
                }),
            )
            .unwrap();
        messages.push(SessionMessage {
            role: SessionMessageRole::Assistant,
            content: output.text.clone(),
        });
        println!("Assistant > {}", output.text.clone());
    }
}
