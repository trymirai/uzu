use std::{env, path::PathBuf};

use uzu::session::{
    ChatSession,
    config::{DecodingConfig, RunConfig},
    types::{Input, Output},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = PathBuf::from(
        env::args().nth(1).expect("Usage: quick_start_chat <MODEL_PATH>"),
    );
    let mut session = ChatSession::new(model_path, DecodingConfig::default())?;

    let input = Input::Text(String::from("Tell about London"));
    let output = session.run(
        input,
        RunConfig::default().tokens_limit(128),
        Some(|_: Output| {
            return true;
        }),
    )?;
    println!("{}", output.text.original);
    Ok(())
}
