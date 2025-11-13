use serde::{Deserialize, Serialize};

use super::{PoolingType, PredictionHeadConfig};
use crate::{
    DecoderConfig, DecoderLayerConfig, EmbeddingConfig, LinearConfig,
    NormalizationConfig, TransformerConfig,
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

impl ClassifierConfig {
    pub fn to_decoder_config(&self) -> DecoderConfig {
        // For classifier, we use the first layer config as the template
        // (most classifiers have homogeneous layers)
        let layer_config = if let Some(first_layer) =
            self.transformer_config.layer_configs.first()
        {
            DecoderLayerConfig {
                pre_attention_norm_config: first_layer
                    .pre_attention_norm_config
                    .clone()
                    .unwrap_or_else(|| {
                        // If first layer has no pre-attention norm (like ModernBERT), use a default
                        self.transformer_config.output_norm_config.clone()
                    }),
                attention_config: first_layer.attention_config.clone(),
                post_attention_norm_config: first_layer
                    .post_attention_norm_config
                    .clone(),
                pre_mlp_norm_config: first_layer.pre_mlp_norm_config.clone(),
                mlp_config: first_layer.mlp_config.clone(),
                post_mlp_norm_config: first_layer.post_mlp_norm_config.clone(),
            }
        } else {
            panic!("TransformerConfig must have at least one layer");
        };

        DecoderConfig {
            embedding_config: self.embedding_config.clone(),
            global_rope_config: self
                .transformer_config
                .global_rope_config
                .clone(),
            local_rope_config: self
                .transformer_config
                .local_rope_config
                .clone(),
            layer_config,
            output_norm_config: self
                .transformer_config
                .output_norm_config
                .clone(),
            vocab_size: self.vocab_size,
            model_dim: self.model_dim,
            hidden_dim: self.transformer_config.hidden_dim,
            num_heads: self.num_heads,
            num_groups: self.num_groups,
            head_dim: self.head_dim,
            attention_scale: self.attention_scale,
            num_layers: self.num_layers,
            sliding_window_sizes: self
                .sliding_window_sizes
                .as_ref()
                .map(|v| v.clone().into_boxed_slice()),
            context_length: self.context_length,
        }
    }
}
