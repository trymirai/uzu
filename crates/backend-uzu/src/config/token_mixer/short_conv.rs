use proc_macros::uzu_config;

use crate::config::{linear::LinearConfig, token_mixer::convolutions::SeparableCausalConvConfig};

#[uzu_config(super::TokenMixerConfig)]
pub struct ShortConvConfig {
    pub in_projection_config: LinearConfig,
    pub conv_config: SeparableCausalConvConfig,
    pub out_projection_config: LinearConfig,

    pub kernel_size: usize,
}
