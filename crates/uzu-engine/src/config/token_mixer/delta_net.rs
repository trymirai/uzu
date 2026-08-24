use uzu_engine_macros::uzu_config;

use crate::config::{
    linear::LinearConfig, normalization::NormalizationConfig, token_mixer::convolutions::SeparableCausalConvConfig,
};

#[uzu_config(super::TokenMixerConfig)]
pub struct DeltaNetConfig {
    pub in_proj_config: LinearConfig,
    pub conv_config: SeparableCausalConvConfig,
    pub out_proj_config: LinearConfig,
    pub norm_config: NormalizationConfig,

    pub num_heads: u32,
    pub num_groups: u32,
    pub head_dim: u32,
    pub value_head_dim: u32,
    pub kernel_size: u32,
}
