use std::path::PathBuf;

use uzu::session::{
    Session,
    config::{DecodingConfig, RunConfig},
    types::{Input, Output},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home =
        std::env::var("HOME").expect("HOME environment variable not set");
    let model_path = PathBuf::from(format!(
        "{}/Developer/trymirai/uzu/models/0.1.6/llama-3.2-1b-instruct",
        home
    ));
    let mut session = Session::new(model_path, DecodingConfig::default())?;

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
