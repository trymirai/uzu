use uzu_engine_macros::uzu_config;

use crate::config::{linear::LinearConfig, normalization::NormalizationConfig, rope::AnyRoPEConfig};

#[uzu_config]
pub struct WeaverConfig {
    pub model_dim: u32,
    pub target_model_dim: u32,
    pub target_embedding_dim: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub hidden_dim: u32,
    pub max_depth: u32,
    pub candidate_pool_size: u32,
    pub linear_config: LinearConfig,
    pub norm_config: NormalizationConfig,
    pub rope_config: AnyRoPEConfig,
}
