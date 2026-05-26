use proc_macros::uzu_config;

use crate::config::{normalization::NormalizationConfig, transformer_layer::TransformerLayerConfig};

#[uzu_config]
pub struct TransformerConfig {
    pub layer_configs: Box<[TransformerLayerConfig]>,
    pub output_norm_config: NormalizationConfig,
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub context_length: usize,
}
