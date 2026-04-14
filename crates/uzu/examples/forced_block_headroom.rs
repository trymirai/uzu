use std::path::PathBuf;

use uzu::session::{
    ChatSession,
    config::{DecodingConfig, RunConfig},
    parameter::{SamplingMethod, SamplingPolicy},
    types::Input,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("UZU_MODEL_PATH").map_err(|_| "UZU_MODEL_PATH environment variable is not set.")?;
    let prompt = std::env::args().nth(1).unwrap_or_else(|| String::from("Tell me one short fact about London."));
    let tokens_limit = std::env::args().nth(2).map(|value| value.parse()).transpose()?.unwrap_or(16_u64);

    let run_config = RunConfig::default().tokens_limit(tokens_limit).sampling_policy(SamplingPolicy::Custom {
        value: SamplingMethod::Greedy,
    });

    let mut baseline_session = ChatSession::new(PathBuf::from(&model_path), DecodingConfig::default())?;
    let (baseline_output, forced_token_path) =
        baseline_session.run_capture_generated_token_ids(Input::Text(prompt.clone()), run_config.clone())?;

    let mut forced_once_session = ChatSession::new(PathBuf::from(&model_path), DecodingConfig::default())?;
    let forced_once_output = forced_once_session.run_forced_token_path_once(
        Input::Text(prompt.clone()),
        run_config.clone(),
        &forced_token_path,
    )?;

    let mut forced_session = ChatSession::new(PathBuf::from(model_path), DecodingConfig::default())?;
    let forced_output = forced_session.run_forced_token_path(
        Input::Text(prompt),
        run_config,
        &forced_token_path,
        None::<fn(uzu::session::types::Output) -> bool>,
    )?;

    let baseline_generate = baseline_output.stats.generate_stats.as_ref().expect("baseline generate stats must exist");
    let forced_once_generate =
        forced_once_output.stats.generate_stats.as_ref().expect("forced-once generate stats must exist");
    let forced_generate = forced_output.stats.generate_stats.as_ref().expect("forced generate stats must exist");

    println!("baseline_text={}", baseline_output.text.original);
    println!("forced_once_text={}", forced_once_output.text.original);
    println!("forced_text={}", forced_output.text.original);
    println!("forced_path_len={}", forced_token_path.len());
    println!("forced_once_output_tokens={}", forced_once_output.stats.total_stats.tokens_count_output);
    println!("forced_once_text_match={}", baseline_output.text.original.starts_with(&forced_once_output.text.original));
    println!("text_match={}", baseline_output.text.original == forced_output.text.original);
    println!("baseline_generate_tps={:.3}", baseline_generate.tokens_per_second);
    println!("forced_once_generate_tps={:.3}", forced_once_generate.tokens_per_second);
    println!("forced_generate_tps={:.3}", forced_generate.tokens_per_second);
    println!("forced_once_speedup={:.3}", forced_once_generate.tokens_per_second / baseline_generate.tokens_per_second);
    println!("speedup={:.3}", forced_generate.tokens_per_second / baseline_generate.tokens_per_second);
    println!("forced_once_speculator_proposed={}", forced_once_generate.speculator_proposed);
    println!("forced_once_speculator_accepted={}", forced_once_generate.speculator_accepted);
    println!("forced_speculator_proposed={}", forced_generate.speculator_proposed);
    println!("forced_speculator_accepted={}", forced_generate.speculator_accepted);

    Ok(())
}
