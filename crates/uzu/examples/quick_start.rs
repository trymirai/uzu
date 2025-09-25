use std::path::PathBuf;

use uzu::session::{
    config::{DecodingConfig, RunConfig},
    parameter::SamplingPolicy,
    session::Session,
    types::{Input, Output},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = PathBuf::from("MODEL_PATH");
    let mut session =
        Session::new(model_path.clone(), DecodingConfig::default())?;

    let input = Input::Text(String::from("Tell about London"));
    let output = session.run(
        input,
        RunConfig::new(128, true, SamplingPolicy::Default),
        Some(|_: Output| {
            return true;
        }),
    )?;
    println!("{}", output.text);
    Ok(())
}
