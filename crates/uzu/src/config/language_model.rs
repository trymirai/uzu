use serde::{Deserialize, Serialize};

use crate::{
    DecoderConfig, DecoderLayerConfig, DecoderLayerType, EmbeddingConfig, GenerationConfig, MessageProcessorConfig,
    MixerConfig, TransformerConfig, config::ConfigError,
};

struct AttentionDims {
    num_heads: usize,
    num_groups: usize,
    head_dim: usize,
    attention_scale: Option<f32>,
}

/// Inner model config matching the new lalamo export format.
/// Contains embedding_config at the top level, with transformer_config nested.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct InnerModelConfig {
    pub embedding_config: EmbeddingConfig,
    pub transformer_config: TransformerConfig,
    pub vocab_size: usize,
}

impl InnerModelConfig {
    /// Convert to DecoderConfig for backward compatibility with the rest of the codebase.
    pub fn to_decoder_config(&self) -> Result<DecoderConfig, ConfigError> {
        let tf = &self.transformer_config;

        let first_layer = tf.layer_configs.first().ok_or(ConfigError::NoLayers)?;

        let layer_config = DecoderLayerConfig {
            pre_attention_norm_config: first_layer
                .pre_attention_norm_config
                .clone()
                .unwrap_or_else(|| tf.output_norm_config.clone()),
            mixer_config: first_layer.mixer_config.clone(),
            post_attention_norm_config: first_layer.post_attention_norm_config.clone(),
            pre_mlp_norm_config: first_layer.pre_mlp_norm_config.clone(),
            mlp_config: first_layer.mlp_config.clone(),
            post_mlp_norm_config: first_layer.post_mlp_norm_config.clone(),
        };

        let attention_dims = Self::derive_attention_dims(tf)?;
        let num_layers = tf.num_layers.unwrap_or(tf.layer_configs.len());

        let sliding_window_sizes: Box<[Option<usize>]> = tf
            .layer_configs
            .iter()
            .map(|layer| layer.mixer_config.sliding_window_size())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let layer_types: Box<[DecoderLayerType]> = tf
            .layer_configs
            .iter()
            .map(|layer| Self::layer_type_from_mixer(&layer.mixer_config))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let layer_configs: Box<[DecoderLayerConfig]> = tf
            .layer_configs
            .iter()
            .map(|layer| DecoderLayerConfig {
                pre_attention_norm_config: layer
                    .pre_attention_norm_config
                    .clone()
                    .unwrap_or_else(|| tf.output_norm_config.clone()),
                mixer_config: layer.mixer_config.clone(),
                post_attention_norm_config: layer.post_attention_norm_config.clone(),
                pre_mlp_norm_config: layer.pre_mlp_norm_config.clone(),
                mlp_config: layer.mlp_config.clone(),
                post_mlp_norm_config: layer.post_mlp_norm_config.clone(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(DecoderConfig {
            embedding_config: self.embedding_config.clone(),
            global_rope_config: tf.global_rope_config.clone(),
            local_rope_config: tf.local_rope_config.clone(),
            layer_config,
            layer_configs: Some(layer_configs),
            output_norm_config: tf.output_norm_config.clone(),
            vocab_size: self.vocab_size,
            model_dim: tf.model_dim,
            hidden_dim: tf.hidden_dim,
            num_heads: attention_dims.num_heads,
            num_groups: attention_dims.num_groups,
            head_dim: attention_dims.head_dim,
            attention_scale: attention_dims.attention_scale,
            num_layers,
            sliding_window_sizes: Some(sliding_window_sizes),
            layer_types: Some(layer_types),
            context_length: tf.context_length,
        })
    }

    fn derive_attention_dims(tf: &TransformerConfig) -> Result<AttentionDims, ConfigError> {
        if let (Some(num_heads), Some(head_dim)) = (tf.num_heads, tf.head_dim) {
            return Ok(AttentionDims {
                num_heads,
                num_groups: tf.num_groups.unwrap_or(num_heads),
                head_dim,
                attention_scale: tf.attention_scale,
            });
        }

        if let Some(attn) = tf.layer_configs.iter().find_map(|layer| layer.mixer_config.as_attention()) {
            let num_heads = attn.num_heads.ok_or_else(|| ConfigError::MissingField("num_heads".to_string()))?;
            let head_dim = attn.head_dim.ok_or_else(|| ConfigError::MissingField("head_dim".to_string()))?;
            return Ok(AttentionDims {
                num_heads,
                num_groups: attn.num_groups.unwrap_or(num_heads),
                head_dim,
                attention_scale: attn.scale,
            });
        }

        if let Some(mamba) = tf.layer_configs.iter().find_map(|layer| layer.mixer_config.as_mamba()) {
            return Ok(AttentionDims {
                num_heads: mamba.num_heads,
                num_groups: mamba.num_groups,
                head_dim: mamba.head_dim,
                attention_scale: None,
            });
        }

        Ok(AttentionDims {
            num_heads: 1,
            num_groups: 1,
            head_dim: tf.model_dim,
            attention_scale: None,
        })
    }

    fn layer_type_from_mixer(mixer: &MixerConfig) -> DecoderLayerType {
        match mixer {
            MixerConfig::Attention(_) => DecoderLayerType::Transformer,
            MixerConfig::Mamba(config) => DecoderLayerType::StateSpace {
                conv_dim: config.conv_dim(),
                kernel_size: config.kernel_size,
                state_dim: config.state_dim,
                num_heads: config.num_heads,
                num_groups: config.num_groups,
                head_dim: config.head_dim,
            },
            MixerConfig::ShortConv(config) => DecoderLayerType::ShortConv {
                kernel_size: config.kernel_size,
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct LanguageModelConfig {
    pub model_config: InnerModelConfig,
    pub message_processor_config: MessageProcessorConfig,
    pub generation_config: GenerationConfig,
}

impl LanguageModelConfig {
    /// Get the decoder config for backward compatibility.
    /// This converts the new format to the old DecoderConfig format.
    pub fn decoder_config(&self) -> Result<DecoderConfig, ConfigError> {
        self.model_config.to_decoder_config()
    }
}
