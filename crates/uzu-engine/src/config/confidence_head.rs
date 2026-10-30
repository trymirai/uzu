use uzu_engine_macros::uzu_config;

use crate::config::{
    confidence_head_tree_attention::ConfidenceHeadTreeAttentionConfig, normalization::NormalizationConfig,
};

#[uzu_config]
pub struct ConfidenceHeadConfig {
    pub model_dim: u32,
    pub pool_hidden_dim: u32,
    pub edge_hidden_dim: u32,
    pub logit_cap: f32,
    pub max_depth: u32,
    pub norm_config: NormalizationConfig,
    pub tree_attention_config: Option<ConfidenceHeadTreeAttentionConfig>,
}
