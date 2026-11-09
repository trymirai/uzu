use uzu_engine_macros::uzu_config;

use crate::config::{
    linear::LinearConfig, normalization::NormalizationConfig, token_mixer::attention::AttentionConfig,
};

#[uzu_config]
pub struct SparseAttentionIndexerConfig {
    pub projection_config: LinearConfig,
    pub query_norm_config: NormalizationConfig,
    pub key_norm_config: NormalizationConfig,
    pub num_heads: u32,
    pub head_dim: u32,
    pub compression_ratio: u32,
    pub token_budget: u32,
}

#[uzu_config(super::TokenMixerConfig)]
pub struct SparseAttentionConfig {
    pub attention_config: AttentionConfig,
    pub indexer_config: SparseAttentionIndexerConfig,
}
