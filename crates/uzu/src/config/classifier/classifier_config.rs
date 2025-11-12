use serde::{Deserialize, Serialize};

use super::{PoolingType, PredictionHeadConfig};
use crate::{
    EmbeddingConfig, LinearConfig, NormalizationConfig, TransformerConfig,
};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ClassifierConfig {
    pub embedding_config: EmbeddingConfig,
    pub embedding_norm_config: NormalizationConfig,
    pub transformer_config: TransformerConfig,
    pub prediction_head_config: PredictionHeadConfig,
    pub final_linear_config: LinearConfig,

    pub vocab_size: usize,
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_groups: usize,
    pub head_dim: usize,
    pub attention_scale: Option<f32>,
    pub num_layers: usize,
    pub sliding_window_sizes: Option<Vec<Option<usize>>>,
    pub context_length: usize,
    pub num_labels: usize,
    pub classifier_pooling: PoolingType,
    pub output_labels: Option<Vec<String>>,
}
