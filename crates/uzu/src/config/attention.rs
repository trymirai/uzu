use serde::{Deserialize, Serialize};

use super::{LinearConfig, NormalizationConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct AttentionConfig {
    pub qkv_projection_config: LinearConfig,
    pub out_projection_config: LinearConfig,

    pub query_norm_config: Option<NormalizationConfig>,
    pub key_norm_config: Option<NormalizationConfig>,

    pub num_heads: Option<usize>,
    pub num_groups: Option<usize>,
    pub head_dim: Option<usize>,
    pub is_causal: Option<bool>,
    pub scale: Option<f32>,
    pub sliding_window_size: Option<usize>,

    pub logit_soft_cap: Option<f32>,
    #[serde(default)]
    pub has_sinks: bool,
    pub has_qkv_biases: bool,
    pub has_out_biases: bool,
    #[serde(default)]
    pub has_gate: bool,
    #[serde(default)]
    pub gate_projection_config: Option<LinearConfig>,
    #[serde(default)]
    pub partial_rope_dim: Option<usize>,
    #[serde(default)]
    pub value_norm_config: Option<NormalizationConfig>,

    /// lalamo PR #197 uses a bool flag instead of full NormalizationConfig.
    /// Consumed during config conversion to populate value_norm_config; not re-serialized.
    #[serde(default, skip_serializing)]
    pub normalize_values: bool,
}

#[cfg(test)]
#[path = "../../tests/unit/config/attention_test.rs"]
mod tests;
