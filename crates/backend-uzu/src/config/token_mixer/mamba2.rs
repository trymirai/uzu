use proc_macros::uzu_config;

use crate::config::{
    activation::AnyActivation, linear::LinearConfig, token_mixer::convolutions::SeparableCausalConvConfig,
};

#[uzu_config(super::TokenMixerConfig)]
pub struct Mamba2Config {
    pub in_projection_config: LinearConfig,
    pub out_projection_config: LinearConfig,
    pub conv_config: SeparableCausalConvConfig,
    pub activation: AnyActivation,

    pub kernel_size: u32,
    pub num_heads: u32,
    pub num_groups: u32,
    pub head_dim: u32,
    pub state_dim: u32,
    pub has_in_biases: bool,
    pub has_out_biases: bool,
}
