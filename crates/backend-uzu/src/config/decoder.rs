use proc_macros::uzu_config;

use crate::config::{
    embedding::AnyEmbeddingConfig, normalization::NormalizationConfig, per_layer_embedding::PLEModelConfig,
    transformer::TransformerConfig,
};

#[uzu_config]
pub struct DecoderConfig {
    pub embedding_config: AnyEmbeddingConfig,
    pub transformer_config: TransformerConfig,

    pub vocab_size: u32,
    pub ple_model_config: Option<PLEModelConfig>,
    pub embedding_norm_config: Option<NormalizationConfig>,
}
