use serde::{Deserialize, Serialize};

use crate::{NormalizationConfig, RoPEConfig, TransformerLayerConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct TransformerConfig {
    pub global_rope_config: RoPEConfig,
    pub local_rope_config: Option<RoPEConfig>,
    pub layer_configs: Vec<TransformerLayerConfig>,
    pub output_norm_config: NormalizationConfig,

    pub model_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_groups: usize,
    pub head_dim: usize,
    pub attention_scale: Option<f32>,
    pub num_layers: usize,
    pub context_length: usize,
}
