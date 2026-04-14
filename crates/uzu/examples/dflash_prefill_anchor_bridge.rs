use std::path::PathBuf;

use uzu::session::{
    ChatSession,
    config::{DecodingConfig, RunConfig},
    parameter::{SamplingMethod, SamplingPolicy},
    types::{Input, Message},
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

    let mut target_session = ChatSession::new(PathBuf::from(&target_model_path), DecodingConfig::default())?;
    let (anchor_output, target_hidden, anchor_token_path) =
        target_session.run_capture_first_generate_step(Input::Text(prompt.clone()), run_config, 5)?;
    assert!(!anchor_token_path.is_empty(), "target prefill must emit at least one token");

    let remaining_tokens = tokens_limit.saturating_sub(anchor_token_path.len() as u64);
    assert!(remaining_tokens > 0, "tokens_limit must exceed the anchored prefix length");
    let continuation_config =
        RunConfig::default().tokens_limit(remaining_tokens).sampling_policy(SamplingPolicy::Custom {
            value: SamplingMethod::Greedy,
        });
    let anchored_input =
        Input::Messages(vec![Message::user(prompt), Message::assistant(anchor_output.text.original.clone(), None)]);

    let mut baseline_session = ChatSession::new(PathBuf::from(&target_model_path), DecodingConfig::default())?;
    let (baseline_output, _) =
        baseline_session.run_capture_generated_token_ids(anchored_input.clone(), continuation_config.clone())?;

    let mut draft_session = ChatSession::new(PathBuf::from(draft_model_path), DecodingConfig::default())?;
    let (draft_output, draft_token_path) =
        draft_session.run_capture_generated_token_ids(anchored_input.clone(), continuation_config.clone())?;

    let mut forced_session = ChatSession::new(PathBuf::from(target_model_path), DecodingConfig::default())?;
    let forced_output =
        forced_session.run_forced_token_path_once(anchored_input, continuation_config, &draft_token_path)?;

    let baseline_generate = baseline_output.stats.generate_stats.as_ref().expect("baseline generate stats must exist");
    let draft_generate = draft_output.stats.generate_stats.as_ref().expect("draft generate stats must exist");
    let forced_generate = forced_output.stats.generate_stats.as_ref().expect("forced generate stats must exist");
    let combined_cycle_tps = forced_generate.tokens_count as f64 / (draft_generate.duration + forced_generate.duration);

    println!("anchor_text={}", anchor_output.text.original);
    println!("baseline_text={}", baseline_output.text.original);
    println!("draft_text={}", draft_output.text.original);
    println!("forced_text={}", forced_output.text.original);
    println!("layer_count={}", target_hidden.layers.len());
    println!("anchor_token_count={}", anchor_token_path.len());
    println!("draft_path_len={}", draft_token_path.len());
    println!("baseline_generate_tps={:.3}", baseline_generate.tokens_per_second);
    println!("draft_generate_tps={:.3}", draft_generate.tokens_per_second);
    println!("forced_generate_tps={:.3}", forced_generate.tokens_per_second);
    println!("combined_cycle_tps={:.3}", combined_cycle_tps);
    println!("verify_only_speedup={:.3}", forced_generate.tokens_per_second / baseline_generate.tokens_per_second);
    println!("combined_cycle_speedup={:.3}", combined_cycle_tps / baseline_generate.tokens_per_second);
    println!("text_match={}", baseline_output.text.original == forced_output.text.original);
    println!("forced_speculator_proposed={}", forced_generate.speculator_proposed);
    println!("forced_speculator_accepted={}", forced_generate.speculator_accepted);

    Ok(())
}
