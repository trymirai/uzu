use proc_macros::uzu_config;

use crate::{
    config::{embedding::AnyEmbeddingConfig, token_mixer::attention::AttentionConfig, transformer::TransformerConfig},
    utils::strict_serde::Unsupported,
};

#[uzu_config]
pub struct DecoderConfig {
    pub embedding_config: AnyEmbeddingConfig,
    pub transformer_config: TransformerConfig,

    pub vocab_size: usize,
    pub pard_token: Option<u64>,
    pub ple_model_config: Option<Unsupported>,
}

impl DecoderConfig {
    pub fn first_attention(&self) -> Option<&AttentionConfig> {
        self.transformer_config.layer_configs.iter().find_map(|layer| layer.mixer_config.as_attention())
    }
}
