use proc_macros::uzu_config;

use super::linear::LinearConfig;
use crate::DataType;

#[uzu_config]
pub struct SeparableCausalConvConfig {
    pub precision: DataType,
    pub has_biases: bool,
}

#[uzu_config]
pub struct ShortConvConfig {
    pub in_projection_config: LinearConfig,
    pub conv_config: SeparableCausalConvConfig,
    pub out_projection_config: LinearConfig,
    pub kernel_size: usize,
}
