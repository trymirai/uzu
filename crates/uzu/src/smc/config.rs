use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Configuration for an SMC-SD session.
///
/// `num_particles` and `ess_threshold` are accepted today but ignored by the
/// Phase-0 serial loop (which hardcodes N=1 and skips resampling).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmcConfig {
    /// Path to the target model folder (expected to contain `config.json`,
    /// `tokenizer.json`, weights).
    pub target_model_path: PathBuf,

    /// Path to the draft model folder. Must share the target's tokenizer
    /// vocabulary — SMC-SD does not support vocabulary projection.
    pub draft_model_path: PathBuf,

    /// Number of particles N. Phase 0: forced to 1.
    #[serde(default = "default_num_particles")]
    pub num_particles: usize,

    /// Draft length γ — how many tokens each particle proposes per step.
    #[serde(default = "default_gamma")]
    pub gamma: usize,

    /// Effective-sample-size threshold below which we fall back to a
    /// target-only greedy step. Fraction of `num_particles`. Phase 0: unused.
    #[serde(default = "default_ess_threshold")]
    pub ess_threshold: f32,

    /// Maximum tokens to emit before stopping (excluding the prompt).
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
}

impl SmcConfig {
    pub fn new(
        target_model_path: PathBuf,
        draft_model_path: PathBuf,
    ) -> Self {
        Self {
            target_model_path,
            draft_model_path,
            num_particles: default_num_particles(),
            gamma: default_gamma(),
            ess_threshold: default_ess_threshold(),
            max_new_tokens: default_max_new_tokens(),
        }
    }
}

fn default_num_particles() -> usize {
    1
}

fn default_gamma() -> usize {
    4
}

fn default_ess_threshold() -> f32 {
    0.5
}

fn default_max_new_tokens() -> usize {
    256
}
