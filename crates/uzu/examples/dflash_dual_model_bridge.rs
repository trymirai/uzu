use std::path::PathBuf;

use uzu::session::{
    ChatSession,
    config::{DecodingConfig, RunConfig},
    parameter::{SamplingMethod, SamplingPolicy},
    types::Input,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target_model_path =
        std::env::var("UZU_TARGET_MODEL_PATH").map_err(|_| "UZU_TARGET_MODEL_PATH environment variable is not set.")?;
    let draft_model_path =
        std::env::var("UZU_DRAFT_MODEL_PATH").map_err(|_| "UZU_DRAFT_MODEL_PATH environment variable is not set.")?;
    let prompt = std::env::args().nth(1).unwrap_or_else(|| String::from("Tell me one short fact about London."));
    let tokens_limit = std::env::args().nth(2).map(|value| value.parse()).transpose()?.unwrap_or(16_u64);

    let run_config = RunConfig::default().tokens_limit(tokens_limit).sampling_policy(SamplingPolicy::Custom {
        value: SamplingMethod::Greedy,
    });

    let mut baseline_session = ChatSession::new(PathBuf::from(&target_model_path), DecodingConfig::default())?;
    let (baseline_output, _) =
        baseline_session.run_capture_generated_token_ids(Input::Text(prompt.clone()), run_config.clone())?;

    let mut target_hidden_session = ChatSession::new(PathBuf::from(&target_model_path), DecodingConfig::default())?;
    let (_target_output, target_hidden) =
        target_hidden_session.run_capture_target_hidden(Input::Text(prompt.clone()), run_config.clone(), 5)?;

    let mut draft_session = ChatSession::new(PathBuf::from(draft_model_path), DecodingConfig::default())?;
    let (draft_output, draft_token_path) =
        draft_session.run_capture_generated_token_ids(Input::Text(prompt.clone()), run_config.clone())?;

    let mut forced_session = ChatSession::new(PathBuf::from(target_model_path), DecodingConfig::default())?;
    let forced_output =
        forced_session.run_forced_token_path_once(Input::Text(prompt), run_config, &draft_token_path)?;

    let baseline_generate = baseline_output.stats.generate_stats.as_ref().expect("baseline generate stats must exist");
    let draft_generate = draft_output.stats.generate_stats.as_ref().expect("draft generate stats must exist");
    let forced_generate = forced_output.stats.generate_stats.as_ref().expect("forced generate stats must exist");
    let combined_cycle_tps = forced_generate.tokens_count as f64 / (draft_generate.duration + forced_generate.duration);
    let verify_only_speedup = forced_generate.tokens_per_second / baseline_generate.tokens_per_second;
    let combined_cycle_speedup = combined_cycle_tps / baseline_generate.tokens_per_second;

    println!("baseline_text={}", baseline_output.text.original);
    println!("draft_text={}", draft_output.text.original);
    println!("forced_text={}", forced_output.text.original);
    println!("layer_count={}", target_hidden.layers.len());
    println!("draft_path_len={}", draft_token_path.len());
    println!("baseline_generate_tps={:.3}", baseline_generate.tokens_per_second);
    println!("draft_generate_tps={:.3}", draft_generate.tokens_per_second);
    println!("forced_generate_tps={:.3}", forced_generate.tokens_per_second);
    println!("combined_cycle_tps={:.3}", combined_cycle_tps);
    println!("verify_only_speedup={:.3}", verify_only_speedup);
    println!("combined_cycle_speedup={:.3}", combined_cycle_speedup);
    println!("text_match={}", baseline_output.text.original == forced_output.text.original);
    println!("forced_speculator_proposed={}", forced_generate.speculator_proposed);
    println!("forced_speculator_accepted={}", forced_generate.speculator_accepted);

    Ok(())
}
