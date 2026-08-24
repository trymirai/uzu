use uzu_engine_macros::uzu_config;

use crate::config::{
    activation::AnyActivation, embedding::AnyEmbeddingConfig, linear::LinearConfig, normalization::NormalizationConfig,
    transformer::TransformerConfig,
};

#[uzu_config]
pub enum PoolingType {
    #[serde(rename = "cls")]
    CLS,
    #[serde(rename = "mean")]
    Mean,
}

#[uzu_config]
pub struct PredictionHeadConfig {
    pub dense_config: LinearConfig,
    pub activation: AnyActivation,
    pub normalization_config: NormalizationConfig,
    pub readout_config: LinearConfig,
    pub use_dense_bias: bool,
}

#[uzu_config]
pub struct ClassifierConfig {
    pub embedding_config: AnyEmbeddingConfig,
    pub embedding_norm_config: NormalizationConfig,
    pub transformer_config: TransformerConfig,
    pub prediction_head_config: PredictionHeadConfig,

    pub vocab_size: u32,
    pub model_dim: u32,
    pub hidden_dim: u32,
    pub num_labels: u32,
    pub classifier_pooling: PoolingType,
    pub output_labels: Option<Box<[String]>>,
}
