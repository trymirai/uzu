use proc_macros::uzu_config;

use crate::config::{linear::LinearConfig, normalization::NormalizationConfig};

#[uzu_config]
pub struct WeaverConfig {
    pub d_model: usize,
    pub d_embed: usize,
    pub d_rank: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub mlp_dim: usize,
    pub k: usize,
    pub candidate_pool_size: usize,
    pub linear_config: LinearConfig,
    pub norm_config: NormalizationConfig,
}
