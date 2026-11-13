use uzu_engine_macros::uzu_config;

use crate::config::{
    linear::LinearConfig, normalization::NormalizationConfig, token_mixer::convolutions::SeparableCausalConvConfig,
};

#[uzu_config]
pub struct NGramHashConfig {
    pub token_vocab_size: u32,
    pub ngram_size: u32,
    pub heads_per_ngram: u32,
    pub eos_token_id: u32,
    pub multipliers: Box<[u64]>,
    pub head_vocab_sizes: Box<[u32]>,
}

#[uzu_config(super::StreamOpConfig)]
pub struct NGramInjectionConfig {
    pub hash_config: NGramHashConfig,
    pub linear_config: LinearConfig,
    pub norm_config: NormalizationConfig,
    pub conv_config: SeparableCausalConvConfig,
    pub embedding_dim: u32,
    pub conv_kernel_size: u32,
}
