use serde::{Deserialize, Serialize};

use crate::config::{
    AttentionConfig, ConfigDataType, ConfigError, DecoderConfig, DecoderLayerConfig, DecoderLayerType, EmbeddingConfig,
    GenerationConfig, MessageProcessorConfig, MixerConfig, NormalizationConfig, TransformerConfig, UpcastMode,
};

struct AttentionDims {
    num_heads: usize,
    num_groups: usize,
    head_dim: usize,
    attention_scale: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct PLEModelConfig {
    #[serde(default)]
    pub ple_dim: Option<usize>,
    #[serde(default)]
    pub ple_embed_scale: Option<f32>,
    #[serde(default)]
    pub model_projection_scale: Option<f32>,
    #[serde(default)]
    pub input_scale: Option<f32>,
    #[serde(default, alias = "linear_config")]
    pub ple_linear_config: Option<crate::config::LinearConfig>,
    #[serde(default, alias = "norm_config")]
    pub ple_norm_config: Option<NormalizationConfig>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct InnerModelConfig {
    pub embedding_config: EmbeddingConfig,
    pub transformer_config: TransformerConfig,
    pub vocab_size: usize,
    #[serde(default)]
    pub hidden_dims: Option<Box<[usize]>>,
    #[serde(default)]
    pub kv_shared_layer_sources: Option<Box<[Option<usize>]>>,
    #[serde(default)]
    pub ple_dim: Option<usize>,
    #[serde(default)]
    pub ple_embed_scale: Option<f32>,
    #[serde(default)]
    pub ple_projection_scale: Option<f32>,
    #[serde(default)]
    pub ple_combination_scale: Option<f32>,
    #[serde(default)]
    pub ple_linear_config: Option<crate::config::LinearConfig>,
    #[serde(default)]
    pub ple_norm_config: Option<NormalizationConfig>,
    #[serde(default)]
    pub has_layer_scalar: bool,

    #[serde(default)]
    pub ple_model_config: Option<PLEModelConfig>,
}

impl InnerModelConfig {
    /// Construct a default V-norm config for `normalize_values == true`.
    fn default_value_norm_config() -> NormalizationConfig {
        NormalizationConfig {
            scale_precision: ConfigDataType::BFloat16,
            accumulation_precision: ConfigDataType::Float32,
            epsilon: 1e-6,
            scale_offset: None,
            upcast_mode: UpcastMode::OnlyNormalization,
            subtract_mean: false,
            use_bias: false,
            has_scale: false,
        }
    }

    fn apply_mixer_conversions(
        mixer: &MixerConfig,
        tf: &TransformerConfig,
    ) -> MixerConfig {
        match mixer {
            MixerConfig::Attention(attn) => MixerConfig::Attention(AttentionConfig {
                value_norm_config: if attn.normalize_values && attn.value_norm_config.is_none() {
                    Some(Self::default_value_norm_config())
                } else {
                    attn.value_norm_config.clone()
                },
                partial_rope_dim: attn.partial_rope_dim.or_else(|| {
                    if attn.sliding_window_size.is_none() {
                        tf.global_rope_dim
                    } else {
                        None
                    }
                }),
                ..attn.clone()
            }),
            other => other.clone(),
        }
    }

    pub fn to_decoder_config(&self) -> Result<DecoderConfig, ConfigError> {
        let tf = &self.transformer_config;
        let ple = self.ple_model_config.as_ref();

        let first_layer = tf.layer_configs.first().ok_or(ConfigError::NoLayers)?;

        let has_layer_scalar = if self.has_layer_scalar {
            true
        } else {
            tf.layer_configs.iter().any(|l| l.ple_config.as_ref().is_some_and(|p| p.has_layer_scalar))
        };

        let first_mixer = Self::apply_mixer_conversions(&first_layer.mixer_config, tf);

        let layer_config = DecoderLayerConfig {
            pre_attention_norm_config: first_layer
                .pre_attention_norm_config
                .clone()
                .unwrap_or_else(|| tf.output_norm_config.clone()),
            mixer_config: first_mixer,
            post_attention_norm_config: first_layer.post_attention_norm_config.clone(),
            pre_mlp_norm_config: first_layer.pre_mlp_norm_config.clone(),
            mlp_config: first_layer.mlp_config.clone(),
            post_mlp_norm_config: first_layer.post_mlp_norm_config.clone(),
            has_layer_scalar,
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
            .map(|layer| {
                let mixer = Self::apply_mixer_conversions(&layer.mixer_config, tf);
                DecoderLayerConfig {
                    pre_attention_norm_config: layer
                        .pre_attention_norm_config
                        .clone()
                        .unwrap_or_else(|| tf.output_norm_config.clone()),
                    mixer_config: mixer,
                    post_attention_norm_config: layer.post_attention_norm_config.clone(),
                    pre_mlp_norm_config: layer.pre_mlp_norm_config.clone(),
                    mlp_config: layer.mlp_config.clone(),
                    post_mlp_norm_config: layer.post_mlp_norm_config.clone(),
                    has_layer_scalar,
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let hidden_dims = self.hidden_dims.clone().or_else(|| {
            let per_layer: Vec<usize> = tf.layer_configs.iter().filter_map(|l| l.hidden_dim).collect();
            if per_layer.len() == tf.layer_configs.len() && !per_layer.is_empty() {
                Some(per_layer.into_boxed_slice())
            } else {
                None
            }
        });

        let kv_shared_layer_sources = self.kv_shared_layer_sources.clone().or_else(|| {
            let has_any = tf.layer_configs.iter().any(|l| l.kv_source_layer.is_some());
            if has_any {
                let sources: Vec<Option<usize>> = tf.layer_configs.iter().map(|l| l.kv_source_layer).collect();
                Some(sources.into_boxed_slice())
            } else {
                None
            }
        });

        let ple_dim = self.ple_dim.or_else(|| ple.and_then(|p| p.ple_dim));
        let ple_embed_scale = self.ple_embed_scale.or_else(|| ple.and_then(|p| p.ple_embed_scale));
        let ple_projection_scale = self.ple_projection_scale.or_else(|| ple.and_then(|p| p.model_projection_scale));
        let ple_combination_scale = self.ple_combination_scale.or_else(|| ple.and_then(|p| p.input_scale));
        let ple_linear_config =
            self.ple_linear_config.clone().or_else(|| ple.and_then(|p| p.ple_linear_config.clone()));
        let ple_norm_config = self.ple_norm_config.clone().or_else(|| ple.and_then(|p| p.ple_norm_config.clone()));

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
            hidden_dims,
            layer_types: Some(layer_types),
            context_length: tf.context_length,
            kv_shared_layer_sources,
            ple_dim,
            ple_embed_scale,
            ple_projection_scale,
            ple_combination_scale,
            ple_linear_config,
            ple_norm_config,
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
    pub fn decoder_config(&self) -> Result<DecoderConfig, ConfigError> {
        self.model_config.to_decoder_config()
    }
}
