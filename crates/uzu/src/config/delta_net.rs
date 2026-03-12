use serde::{Deserialize, Serialize};

use super::{linear::LinearConfig, normalization::NormalizationConfig, short_conv::SeparableCausalConvConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DeltaNetAttentionConfig {
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

impl DeltaNetAttentionConfig {
    pub fn key_dim(&self) -> usize {
        self.num_groups * self.head_dim
    }

    pub fn value_dim(&self) -> usize {
        self.num_heads * self.value_head_dim
    }

    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
}
