use proc_macros::uzu_config;

use crate::config::{
    embedding::AnyEmbeddingConfig, per_layer_embedding::PLEModelConfig, token_mixer::attention::AttentionConfig,
    transformer::TransformerConfig,
};

#[uzu_config]
pub struct DecoderConfig {
    pub embedding_config: AnyEmbeddingConfig,
    pub transformer_config: TransformerConfig,

    pub vocab_size: usize,
    pub ple_model_config: Option<PLEModelConfig>,
}

impl DecoderConfig {
    pub fn first_attention(&self) -> Option<&AttentionConfig> {
        self.transformer_config.layer_configs.iter().find_map(|layer| layer.mixer_config.as_attention())
    }
}
