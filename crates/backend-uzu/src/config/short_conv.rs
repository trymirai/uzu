use proc_macros::uzu_config;

use super::{common::ConfigDataType, linear::LinearConfig};

#[uzu_config]
pub struct SeparableCausalConvConfig {
    pub precision: ConfigDataType,
    pub has_biases: bool,
}

#[uzu_config]
pub struct ShortConvConfig {
    pub in_projection_config: LinearConfig,
    pub conv_config: SeparableCausalConvConfig,
    pub out_projection_config: LinearConfig,
    pub kernel_size: usize,
}
