use uzu_engine_macros::uzu_config;

use crate::config::{convolutions::SeparableCausalConvConfig, linear::LinearConfig};

#[uzu_config(super::TokenMixerConfig)]
pub struct ShortConvConfig {
    pub in_projection_config: LinearConfig,
    pub conv_config: SeparableCausalConvConfig,
    pub out_projection_config: LinearConfig,

    pub kernel_size: u32,
}
