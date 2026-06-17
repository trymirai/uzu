use proc_macros::uzu_config;

use crate::config::{normalization::NormalizationConfig, transformer_layer::TransformerLayerConfig};

#[uzu_config]
pub struct TransformerConfig {
    pub layer_configs: Box<[TransformerLayerConfig]>,
    pub output_norm_config: NormalizationConfig,
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub kv_source_per_layer: Option<Box<[usize]>>,
}

impl TransformerConfig {
    pub fn kv_source_layer_index(
        &self,
        layer_index: usize,
    ) -> Option<usize> {
        self.kv_source_per_layer
            .as_ref()
            .map(|kv_source_per_layer| kv_source_per_layer[layer_index])
            .or(self.layer_configs[layer_index].kv_source_layer_index)
            .filter(|&source| source != layer_index)
    }

    pub fn kv_source_layer_indices(&self) -> Box<[Option<usize>]> {
        (0..self.layer_configs.len()).map(|layer_index| self.kv_source_layer_index(layer_index)).collect()
    }

    pub fn max_sequence_length(&self) -> Option<usize> {
        self.layer_configs
            .iter()
            .filter_map(|layer_config| {
                layer_config.rope_config.as_ref().map(|rope_config| *rope_config.max_sequence_length())
            })
            .max()
    }
}
