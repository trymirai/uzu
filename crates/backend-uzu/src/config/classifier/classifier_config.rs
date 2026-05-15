use proc_macros::uzu_config;

use super::{PoolingType, PredictionHeadConfig};
use crate::config::{EmbeddingConfig, LinearConfig, NormalizationConfig, TransformerConfig};

#[uzu_config]
pub struct ClassifierConfig {
    pub embedding_config: EmbeddingConfig,
    pub embedding_norm_config: NormalizationConfig,
    pub transformer_config: TransformerConfig,
    pub prediction_head_config: PredictionHeadConfig,
    pub readout_config: LinearConfig,
    pub vocab_size: usize,
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub attention_scale: Option<f32>,
    pub num_layers: usize,
    pub context_length: usize,
    pub num_labels: usize,
    pub classifier_pooling: PoolingType,
    pub output_labels: Option<Vec<String>>,
}
