use uzu_engine_macros::uzu_config;

use crate::config::{linear::LinearConfig, normalization::NormalizationConfig};

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum MarkovHeadType {
    RnnHidden,
}

#[uzu_config(super::DraftAdapterConfig)]
pub struct DSparkConfig {
    pub model_dim: u32,
    pub vocab_size: u32,
    pub num_context_taps: u32,
    pub markov_rank: u32,
    pub markov_head_type: MarkovHeadType,
    pub projection_config: LinearConfig,
    pub norm_config: NormalizationConfig,
    pub logit_soft_cap: Option<f32>,
}
