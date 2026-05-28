use proc_macros::uzu_config;

use crate::config::{normalization::NormalizationConfig, transformer_layer::TransformerLayerConfig};

#[uzu_config]
pub struct TransformerConfig {
    pub layer_configs: Box<[TransformerLayerConfig]>,
    pub output_norm_config: NormalizationConfig,
    pub model_dim: usize,
    pub hidden_dim: usize,
}

impl TransformerConfig {
    pub fn max_sequence_length(&self) -> Option<usize> {
        self.layer_configs
            .iter()
            .filter_map(|layer_config| {
                layer_config.rope_config.as_ref().map(|rope_config| *rope_config.max_sequence_length())
            })
            .max()
    }
}
