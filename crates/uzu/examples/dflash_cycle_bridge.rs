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

    let run_config = RunConfig::default().tokens_limit(16).sampling_policy(SamplingPolicy::Custom {
        value: SamplingMethod::Greedy,
    });

    let mut oracle_session = ChatSession::new(PathBuf::from(&model_path), DecodingConfig::default())?;
    let (oracle_output, target_hidden, forced_token_path) =
        oracle_session.run_capture_first_generate_step(Input::Text(prompt.clone()), run_config.clone(), 5)?;

    let mut forced_session = ChatSession::new(PathBuf::from(model_path), DecodingConfig::default())?;
    let forced_output =
        forced_session.run_forced_token_path_once(Input::Text(prompt), run_config, &forced_token_path)?;

    let oracle_prefill = &oracle_output.stats.prefill_stats;
    let forced_generate = forced_output.stats.generate_stats.as_ref().expect("forced generate stats must exist");

    println!("oracle_text={}", oracle_output.text.original);
    println!("forced_text={}", forced_output.text.original);
    println!("layer_count={}", target_hidden.layers.len());
    println!("forced_path_len={}", forced_token_path.len());
    println!("oracle_prefill_tps={:.3}", oracle_prefill.tokens_per_second);
    println!("forced_generate_tps={:.3}", forced_generate.tokens_per_second);
    println!("speedup={:.3}", forced_generate.tokens_per_second / oracle_prefill.tokens_per_second);
    println!("forced_speculator_proposed={}", forced_generate.speculator_proposed);
    println!("forced_speculator_accepted={}", forced_generate.speculator_accepted);

    Ok(())
}
