//! Phase-0 smoke test for the SMC-SD scaffolding.
//!
//! Loads a target + draft model pair via `SmcSession` and runs a single
//! generation on the target. Does NOT yet exercise the draft for speculation —
//! that's Phase 2. This binary exists to prove the two-model loading path
//! works end-to-end.
//!
//! Usage:
//!   UZU_SMC_TARGET=/path/to/target-model \
//!   UZU_SMC_DRAFT=/path/to/draft-model  \
//!   cargo run --example smc_demo --release

use std::path::PathBuf;

use uzu::{
    session::{
        config::RunConfig,
        types::{Input, Output},
    },
    smc::{SmcConfig, SmcSession},
};

fn env_path(name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let val = std::env::var(name)
        .map_err(|_| -> Box<dyn std::error::Error> { format!("{name} is not set").into() })?;
    Ok(PathBuf::from(val))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = env_path("UZU_SMC_TARGET")?;
    let draft = env_path("UZU_SMC_DRAFT")?;

    let cfg = SmcConfig::new(target, draft);
    println!(
        "[smc_demo] target={}",
        cfg.target_model_path.display()
    );
    println!(
        "[smc_demo] draft ={}",
        cfg.draft_model_path.display()
    );
    println!(
        "[smc_demo] N={} gamma={} (phase 0: values ignored, target-only emit)",
        cfg.num_particles, cfg.gamma
    );

    let mut session = SmcSession::new(cfg)?;

    let prompt = std::env::var("UZU_SMC_PROMPT").unwrap_or_else(|_| "Tell about London".to_string());
    let output = session.generate(
        Input::Text(prompt),
        RunConfig::default().tokens_limit(128),
        Some(|_: Output| true),
    )?;
    println!("{}", output.text.original);
    Ok(())
}
