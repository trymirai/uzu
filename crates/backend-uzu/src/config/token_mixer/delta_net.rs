use proc_macros::uzu_config;

use crate::config::{
    linear::LinearConfig, normalization::NormalizationConfig, token_mixer::convolutions::SeparableCausalConvConfig,
};

#[uzu_config(super::TokenMixerConfig)]
pub struct DeltaNetConfig {
    pub in_proj_config: LinearConfig,
    pub conv_config: SeparableCausalConvConfig,
    pub out_proj_config: LinearConfig,
    pub norm_config: NormalizationConfig,

    pub num_heads: usize,
    pub num_groups: usize,
    pub head_dim: usize,
    pub value_head_dim: usize,
    pub kernel_size: usize,
}
