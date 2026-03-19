use serde::{Deserialize, Serialize};

use crate::{LinearConfig, NormalizationConfig};

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
}

#[cfg(test)]
#[path = "../../tests_unit/config/attention_test.rs"]
mod tests;
