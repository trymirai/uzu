use serde::{Deserialize, Serialize};

use super::{
    common::{Activation, ConfigDataType},
    linear::LinearConfig,
};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct CausalConv1dConfig {
    pub kernel_size: usize,
    pub activation: Activation,
    pub precision: ConfigDataType,
    pub has_biases: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Mamba2Config {
    pub in_projection_config: LinearConfig,
    pub out_projection_config: LinearConfig,
    pub conv_config: CausalConv1dConfig,
    pub num_value_heads: usize,
    pub num_groups: usize,
    pub head_dim: usize,
    pub state_dim: usize,
    pub expand: usize,
    pub has_in_biases: bool,
    pub has_out_biases: bool,
}

impl Mamba2Config {
    pub fn inner_dim(&self) -> usize {
        self.num_value_heads * self.head_dim
    }

    pub fn conv_dim(&self) -> usize {
        self.inner_dim() + 2 * self.num_groups * self.state_dim
    }
}
