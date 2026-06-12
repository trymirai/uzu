use proc_macros::uzu_config;

use super::{activation::AnyActivation, linear::LinearConfig, normalization::NormalizationConfig};

#[uzu_config]
pub struct PLEModelConfig {
    pub ple_dim: usize,
    pub num_layers: usize,
    pub ple_vocab_size: usize,
    pub ple_embed_scale: f32,
    pub model_projection_scale: f32,
    pub input_scale: f32,
    pub linear_config: LinearConfig,
    pub norm_config: NormalizationConfig,
}

#[uzu_config]
pub struct PLELayerConfig {
    pub linear_config: LinearConfig,
    pub norm_config: NormalizationConfig,
    pub ple_dim: usize,
    pub activation: AnyActivation,
}
