mod common;
use std::path::PathBuf;

use uzu::session::{
    session::Session,
    session_config::{SessionConfig, SessionRunConfig},
    session_input::SessionInput,
    session_output::SessionOutput,
};

fn build_model_path() -> PathBuf {
    common::get_test_model_path()
}

#[test]
fn test_generation_base() {
    let text = String::from("Tell about London");
    let config = SessionConfig::default();
    run(text, config, 128);
}

fn run(
    text: String,
    config: SessionConfig,
    tokens_limit: u64,
) {
    let mut session = Session::new(build_model_path()).unwrap();
    session.load_with_session_config(config).unwrap();

    let input = SessionInput::Text(text);
    let output = session.run(
        input,
        SessionRunConfig::new(tokens_limit),
        Some(|_: SessionOutput| {
            return true;
        }),
    );

    println!("-------------------------");
    println!("{}", output.text);
    println!("-------------------------");
    println!("{:#?}", output.stats);
    println!("-------------------------");
    println!("Finish reason: {:?}", output.finish_reason);
    println!("-------------------------");
}
