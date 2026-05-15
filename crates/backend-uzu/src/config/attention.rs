use serde::{Deserialize, Serialize};

use super::{LinearConfig, NormalizationConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct AttentionConfig {
    pub qkv_projection_config: LinearConfig,
    pub out_projection_config: LinearConfig,

    pub query_norm_config: Option<NormalizationConfig>,
    pub key_norm_config: Option<NormalizationConfig>,
    pub value_norm_config: Option<NormalizationConfig>,

    pub num_heads: usize,
    pub num_groups: usize,
    pub head_dim: usize,
    pub is_causal: bool,
    pub scale: Option<f32>,
    pub sliding_window_size: Option<usize>,

    pub logit_soft_cap: Option<f32>,
    pub has_sinks: bool,
    pub has_qkv_biases: bool,
    pub has_out_biases: bool,
    pub gate_projection_config: Option<LinearConfig>,
}
