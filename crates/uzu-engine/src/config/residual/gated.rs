use uzu_engine_macros::uzu_config;

use crate::config::{linear::LinearConfig, normalization::NormalizationConfig};

#[uzu_config(super::ResidualConfig)]
pub struct GatedResidualConfig {
    pub stream_count: u32,
    pub low_rank_dim: u32,
    pub norm_config: NormalizationConfig,
    pub linear_config: LinearConfig,
}
