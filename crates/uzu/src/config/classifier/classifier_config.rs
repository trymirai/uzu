use serde::{Deserialize, Serialize};

use super::{PoolingType, PredictionHeadConfig};
use crate::{
    DecoderConfig, DecoderLayerConfig, EmbeddingConfig, LinearConfig,
    NormalizationConfig, TransformerConfig, config::ConfigError,
};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ClassifierConfig {
    pub embedding_config: EmbeddingConfig,
    pub embedding_norm_config: NormalizationConfig,
    pub transformer_config: TransformerConfig,
    pub prediction_head_config: PredictionHeadConfig,
    pub readout_config: LinearConfig,

    pub vocab_size: usize,
    pub model_dim: usize,
    pub hidden_dim: usize,
    #[serde(default)]
    pub num_heads: Option<usize>,
    #[serde(default)]
    pub num_groups: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub attention_scale: Option<f32>,
    pub num_layers: usize,
    #[serde(default)]
    pub sliding_window_sizes: Option<Vec<Option<usize>>>,
    pub context_length: usize,
    pub num_labels: usize,
    pub classifier_pooling: PoolingType,
    pub output_labels: Option<Vec<String>>,
}

impl ClassifierConfig {
    pub fn to_decoder_config(&self) -> Result<DecoderConfig, ConfigError> {
        // For classifier, we use the first layer config as the template
        // (most classifiers have homogeneous layers)
        let first_layer = self
            .transformer_config
            .layer_configs
            .first()
            .ok_or(ConfigError::NoLayers)?;

        let layer_config = DecoderLayerConfig {
            pre_attention_norm_config: first_layer
                .pre_attention_norm_config
                .clone()
                .unwrap_or_else(|| {
                    // If first layer has no pre-attention norm (like ModernBERT), use a default
                    self.transformer_config.output_norm_config.clone()
                }),
            mixer_config: first_layer.mixer_config.clone(),
            post_attention_norm_config: first_layer
                .post_attention_norm_config
                .clone(),
            pre_mlp_norm_config: first_layer.pre_mlp_norm_config.clone(),
            mlp_config: first_layer.mlp_config.clone(),
            post_mlp_norm_config: first_layer.post_mlp_norm_config.clone(),
        };

        // Derive head parameters from either top-level config or first layer's mixer
        let first_mixer = &first_layer.mixer_config;

        let num_heads =
            self.num_heads.or(first_mixer.num_heads()).ok_or_else(|| {
                ConfigError::MissingField("num_heads".to_string())
            })?;
        let num_groups =
            self.num_groups.or(first_mixer.num_groups()).unwrap_or(num_heads);
        let head_dim = self
            .head_dim
            .or(first_mixer.head_dim())
            .ok_or_else(|| ConfigError::MissingField("head_dim".to_string()))?;

        Ok(DecoderConfig {
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
            num_heads,
            num_groups,
            head_dim,
            attention_scale: self.attention_scale,
            num_layers: self.num_layers,
            sliding_window_sizes: {
                // If sliding_window_sizes is explicitly provided, use it
                // Otherwise, extract from per-layer mixer configs
                if let Some(sizes) = &self.sliding_window_sizes {
                    Some(sizes.clone().into_boxed_slice())
                } else {
                    // Extract sliding_window_size from each layer's mixer_config
                    let sizes: Vec<Option<usize>> = self
                        .transformer_config
                        .layer_configs
                        .iter()
                        .map(|layer| layer.mixer_config.sliding_window_size())
                        .collect();
                    Some(sizes.into_boxed_slice())
                }
            },
            context_length: self.context_length,
            layer_configs: None, // Classifier doesn't use heterogeneous layers usually
            layer_types: None,
        })
    }
}
