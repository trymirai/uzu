use serde::{Deserialize, Serialize};

use super::linear::LinearConfig;
use crate::DataType;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct SeparableCausalConvConfig {
    pub precision: DataType,
    pub has_biases: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ShortConvConfig {
    pub in_projection_config: LinearConfig,
    pub conv_config: SeparableCausalConvConfig,
    pub out_projection_config: LinearConfig,
    pub kernel_size: usize,
}
