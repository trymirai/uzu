use std::path::PathBuf;

use uzu::session::{
    ChatSession,
    config::{DecodingConfig, RunConfig},
    types::{Input, Output},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("UZU_MODEL_PATH")
        .map_err(|_| -> Box<dyn std::error::Error> { "UZU_MODEL_PATH environment variable is not set.".into() })?;
    let model_path_buf = PathBuf::from(model_path);
    let mut session = ChatSession::new(model_path_buf, DecodingConfig::default())?;

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
