use serde::{Deserialize, Serialize};

use crate::config::{
    ConfigError, DecoderConfig, DecoderLayerConfig, DecoderLayerType, EmbeddingConfig, GenerationConfig,
    MessageProcessorConfig, MixerConfig, PLEModelConfig, TransformerConfig, resolve_rope_configs,
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
    #[serde(default)]
    pub ple_model_config: Option<PLEModelConfig>,
}

impl InnerModelConfig {
    pub fn new(
        embedding_config: EmbeddingConfig,
        transformer_config: TransformerConfig,
        vocab_size: usize,
    ) -> Self {
        Self {
            embedding_config,
            transformer_config,
            vocab_size,
            ple_model_config: None,
        }
    }

    pub fn with_ple_model_config(
        mut self,
        ple_model_config: Option<PLEModelConfig>,
    ) -> Self {
        self.ple_model_config = ple_model_config;
        self
    }

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
            hidden_dim: first_layer.hidden_dim,
            ple_config: first_layer.ple_config.clone(),
            has_post_layer_scalar: first_layer.has_post_layer_scalar,
            kv_source_layer: first_layer.kv_source_layer,
            rope_config: first_layer.rope_config.clone(),
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
                hidden_dim: layer.hidden_dim,
                ple_config: layer.ple_config.clone(),
                has_post_layer_scalar: layer.has_post_layer_scalar,
                kv_source_layer: layer.kv_source_layer,
                rope_config: layer.rope_config.clone(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let (global_rope_config, local_rope_config) = resolve_rope_configs(
            tf.global_rope_config.clone(),
            tf.local_rope_config.clone(),
            &layer_config,
            Some(&layer_configs),
        )
        .map_err(ConfigError::Invalid)?;

        Ok(DecoderConfig {
            embedding_config: self.embedding_config.clone(),
            global_rope_config,
            local_rope_config,
            ple_model_config: self.ple_model_config.clone(),
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

        let mut attention_dims: Option<AttentionDims> = None;
        for attn in tf.layer_configs.iter().filter_map(|layer| layer.mixer_config.as_attention()) {
            let num_heads = attn.num_heads.ok_or_else(|| ConfigError::MissingField("num_heads".to_string()))?;
            let head_dim = attn.head_dim.ok_or_else(|| ConfigError::MissingField("head_dim".to_string()))?;
            let num_groups = attn.num_groups.unwrap_or(num_heads);
            attention_dims = Some(match attention_dims {
                Some(current) => AttentionDims {
                    num_heads: current.num_heads.max(num_heads),
                    num_groups: current.num_groups.max(num_groups),
                    head_dim: current.head_dim.max(head_dim),
                    attention_scale: current.attention_scale.or(attn.scale),
                },
                None => AttentionDims {
                    num_heads,
                    num_groups,
                    head_dim,
                    attention_scale: attn.scale,
                },
            });
        }
        if let Some(attention_dims) = attention_dims {
            return Ok(attention_dims);
        }

        if let Some(mamba) = tf.layer_configs.iter().find_map(|layer| layer.mixer_config.as_mamba()) {
            return Ok(AttentionDims {
                num_heads: mamba.num_heads,
                num_groups: mamba.num_groups,
                head_dim: mamba.head_dim,
                attention_scale: None,
            });
        }

        if let Some(dn) = tf.layer_configs.iter().find_map(|layer| layer.mixer_config.as_delta_net()) {
            return Ok(AttentionDims {
                num_heads: dn.num_heads,
                num_groups: dn.num_groups,
                head_dim: dn.head_dim,
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
            MixerConfig::DeltaNet(config) => DecoderLayerType::DeltaNet {
                conv_dim: config.conv_dim(),
                kernel_size: config.kernel_size,
                num_heads: config.num_heads,
                num_groups: config.num_groups,
                head_dim: config.head_dim,
                value_head_dim: config.value_head_dim,
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
