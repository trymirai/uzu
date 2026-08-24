use uzu_engine_macros::uzu_config;

use crate::config::{normalization::NormalizationConfig, transformer_layer::TransformerLayerConfig};

#[uzu_config]
pub struct TransformerConfig {
    pub layer_configs: Box<[TransformerLayerConfig]>,
    pub output_norm_config: NormalizationConfig,
    pub model_dim: u32,
    pub hidden_dim: u32,
}
