use uzu_engine_macros::uzu_config;

use crate::config::{activation::AnyActivation, linear::LinearConfig, normalization::NormalizationConfig};

#[uzu_config(super::StreamOpConfig)]
pub struct PerLayerEmbeddingInjectionConfig {
    pub linear_config: LinearConfig,
    pub norm_config: NormalizationConfig,
    pub ple_dim: u32,
    pub activation: AnyActivation,
}
