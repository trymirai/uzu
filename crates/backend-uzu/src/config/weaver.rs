use proc_macros::uzu_config;

use crate::config::{linear::LinearConfig, normalization::NormalizationConfig};

#[uzu_config]
pub struct WeaverConfig {
    pub model_dim: usize,
    pub target_model_dim: usize,
    pub target_embedding_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub hidden_dim: usize,
    pub max_depth: usize,
    pub candidate_pool_size: usize,
    pub linear_config: LinearConfig,
    pub norm_config: NormalizationConfig,
}
