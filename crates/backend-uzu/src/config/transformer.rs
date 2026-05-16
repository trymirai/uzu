use proc_macros::uzu_config;

use super::{NormalizationConfig, TransformerLayerConfig};

#[uzu_config]
pub struct TransformerConfig {
    pub layer_configs: Vec<TransformerLayerConfig>,
    pub output_norm_config: NormalizationConfig,
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub context_length: usize,
}
