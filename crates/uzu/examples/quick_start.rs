use std::path::PathBuf;

use uzu::session::{
    ChatSession,
    config::{DecodingConfig, RunConfig},
    types::{Input, Output},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| {
            String::from("models/0.1.6/Llama-3.2-1B-Instruct")
        }));
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
