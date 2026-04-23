use serde::{Deserialize, Serialize};

use crate::{
    backends::common::ActivationConfig,
    config::{LinearConfig, NormalizationConfig},
};

/// Config for a classifier's prediction head.
///
/// The head is a chain `maybe(dense) → maybe(activation) → maybe(norm) → readout`.
/// BERT-style heads set all four; bare linear token taggers (e.g.
/// openai/privacy-filter) set only `readout_config` and leave the rest `None`.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct PredictionHeadConfig {
    /// Final linear projection to `num_labels`. Always present.
    pub readout_config: LinearConfig,
    /// Optional pre-readout projection (same dimension as model_dim).
    #[serde(default)]
    pub dense_config: Option<LinearConfig>,
    /// Optional activation applied after `dense`.
    #[serde(default)]
    pub activation: Option<ActivationConfig>,
    /// Optional normalization applied after the activation.
    #[serde(default)]
    pub normalization_config: Option<NormalizationConfig>,
    /// Whether the optional `dense` linear carries a bias. Ignored if
    /// `dense_config` is `None`.
    #[serde(default)]
    pub use_dense_bias: bool,
    /// Whether the final `readout` linear carries a bias. Defaults to `true`
    /// because that's what lalamo emits for classifier heads (and the
    /// privacy-filter spec).
    #[serde(default = "default_readout_has_biases")]
    pub readout_has_biases: bool,
}

fn default_readout_has_biases() -> bool {
    true
}
