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
    #[serde(default)]
    pub num_heads: Option<usize>,
    #[serde(default)]
    pub num_groups: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub attention_scale: Option<f32>,
    #[serde(default)]
    pub num_layers: Option<usize>,
    pub context_length: usize,
}
