use uzu_engine_macros::uzu_config;

use super::{linear::LinearConfig, normalization::NormalizationConfig};

#[uzu_config]
pub struct PLEModelConfig {
    pub ple_dim: u32,
    pub num_layers: u32,
    pub ple_vocab_size: u32,
    pub ple_embed_scale: f32,
    pub model_projection_scale: f32,
    pub input_scale: f32,
    pub linear_config: LinearConfig,
    pub norm_config: NormalizationConfig,
}
