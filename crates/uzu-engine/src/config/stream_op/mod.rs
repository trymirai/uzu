use uzu_engine_macros::uzu_config_abstract;

pub mod per_layer_embedding_injection;
pub mod residual_scale;

#[uzu_config_abstract(
    per_layer_embedding_injection::PerLayerEmbeddingInjectionConfig,
    residual_scale::ResidualScaleConfig
)]
pub struct StreamOpConfig;
