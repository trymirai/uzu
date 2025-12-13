use serde::{
    Deserialize, Serialize,
    de::{self, Deserializer},
};

use super::{
    decoder_layer::{DecoderLayerConfig, MixerConfig},
    embedding::EmbeddingConfig,
    normalization::NormalizationConfig,
    rope::RoPEConfig,
};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecoderLayerType {
    Transformer,
    #[serde(rename = "ssm")]
    StateSpace {
        conv_dim: usize,
        kernel_size: usize,
        state_dim: usize,
        num_heads: usize,
        num_groups: usize,
        head_dim: usize,
    },
    #[serde(rename = "short_conv")]
    ShortConv {
        kernel_size: usize,
    },
}

#[derive(Debug, Serialize, PartialEq, Clone)]
pub struct DecoderConfig {
    pub embedding_config: EmbeddingConfig,
    pub global_rope_config: Option<RoPEConfig>,
    pub local_rope_config: Option<RoPEConfig>,
    pub layer_config: DecoderLayerConfig,
    pub layer_configs: Option<Box<[DecoderLayerConfig]>>,
    pub output_norm_config: NormalizationConfig,

    pub vocab_size: usize,
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_groups: usize,
    pub head_dim: usize,
    pub attention_scale: Option<f32>,
    pub num_layers: usize,
    pub sliding_window_sizes: Option<Box<[Option<usize>]>>,
    pub layer_types: Option<Box<[DecoderLayerType]>>,
    pub context_length: usize,
}

impl<'de> Deserialize<'de> for DecoderConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawDecoderConfig::deserialize(deserializer)?;
        let RawDecoderConfig {
            embedding_config,
            global_rope_config,
            local_rope_config,
            layer_config,
            layer_configs,
            output_norm_config,
            vocab_size,
            model_dim,
            hidden_dim,
            num_heads,
            num_groups,
            head_dim,
            attention_scale,
            num_layers,
            sliding_window_sizes,
            layer_types,
            context_length,
        } = raw;

        let layer_configs_boxed =
            layer_configs.map(|layers| layers.into_boxed_slice());

        let layer_config_value = if let Some(config) = layer_config {
            config
        } else if let Some(configs) = layer_configs_boxed.as_ref() {
            configs.first().cloned().ok_or_else(|| {
                de::Error::custom("layer_configs must not be empty")
            })?
        } else {
            return Err(de::Error::custom(
                "decoder config must include layer_config or layer_configs",
            ));
        };

        let num_layers_value = if let Some(value) = num_layers {
            value
        } else if let Some(configs) = layer_configs_boxed.as_ref() {
            configs.len()
        } else {
            return Err(de::Error::custom(
                "num_layers missing and layer_configs not provided",
            ));
        };

        let (num_heads_value, num_groups_value, head_dim_value) =
            match (num_heads, num_groups, head_dim) {
                (Some(h), Some(g), Some(d)) => (h, g, d),
                _ => derive_dims_from_layer(&layer_config_value).ok_or_else(
                    || {
                        de::Error::custom(
                            "num_heads/num_groups/head_dim missing and \
                             cannot be derived from layer config",
                        )
                    },
                )?,
            };

        let attention_scale_value = if attention_scale.is_some() {
            attention_scale
        } else {
            layer_config_value.mixer_config.attention_scale()
        };

        let sliding_window_sizes_boxed =
            if let Some(sizes) = sliding_window_sizes {
                Some(sizes.into_boxed_slice())
            } else if let Some(configs) = layer_configs_boxed.as_ref() {
                Some(
                    configs
                        .iter()
                        .map(|layer| layer.mixer_config.sliding_window_size())
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )
            } else {
                None
            };

        let explicit_layer_types =
            layer_types.map(|types| types.into_boxed_slice());
        let derived_layer_types =
            if let Some(configs) = layer_configs_boxed.as_ref() {
                Some(
                    configs
                        .iter()
                        .map(layer_type_from_config)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )
            } else {
                None
            };
        let layer_types_value = explicit_layer_types.or(derived_layer_types);

        Ok(Self {
            embedding_config,
            global_rope_config,
            local_rope_config,
            layer_config: layer_config_value,
            layer_configs: layer_configs_boxed,
            output_norm_config,
            vocab_size,
            model_dim,
            hidden_dim,
            num_heads: num_heads_value,
            num_groups: num_groups_value,
            head_dim: head_dim_value,
            attention_scale: attention_scale_value,
            num_layers: num_layers_value,
            sliding_window_sizes: sliding_window_sizes_boxed,
            layer_types: layer_types_value,
            context_length,
        })
    }
}

#[derive(Deserialize)]
struct RawDecoderConfig {
    embedding_config: EmbeddingConfig,
    #[serde(default)]
    global_rope_config: Option<RoPEConfig>,
    #[serde(default)]
    local_rope_config: Option<RoPEConfig>,
    #[serde(default)]
    layer_config: Option<DecoderLayerConfig>,
    #[serde(default)]
    layer_configs: Option<Vec<DecoderLayerConfig>>,
    output_norm_config: NormalizationConfig,
    vocab_size: usize,
    model_dim: usize,
    hidden_dim: usize,
    #[serde(default)]
    num_heads: Option<usize>,
    #[serde(default)]
    num_groups: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    attention_scale: Option<f32>,
    #[serde(default)]
    num_layers: Option<usize>,
    #[serde(default)]
    sliding_window_sizes: Option<Vec<Option<usize>>>,
    #[serde(default)]
    layer_types: Option<Vec<DecoderLayerType>>,
    context_length: usize,
}

fn derive_dims_from_layer(
    layer: &DecoderLayerConfig
) -> Option<(usize, usize, usize)> {
    Some((
        layer.mixer_config.num_heads()?,
        layer.mixer_config.num_groups()?,
        layer.mixer_config.head_dim()?,
    ))
}

fn layer_type_from_config(layer: &DecoderLayerConfig) -> DecoderLayerType {
    match &layer.mixer_config {
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
impl DecoderConfig {
    pub fn group_size(&self) -> usize {
        self.num_heads * self.num_groups
    }

    pub fn has_attention_layers(&self) -> bool {
        self.layer_configs
            .as_ref()
            .map(|configs| {
                configs.iter().any(|config| {
                    matches!(
                        config.mixer_config,
                        crate::config::MixerConfig::Attention(_)
                    )
                })
            })
            .unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use serde_json5::from_str;

    use super::{
        super::{
            attention::AttentionConfig,
            common::{Activation, ConfigDataType, QuantizationMode},
            embedding::{EmbeddingConfig, EmbeddingConfigCommon},
            linear::{LinearConfig, QuantizationConfig},
            mlp::{DenseMLPConfig, MLPConfig},
            normalization::UpcastMode,
            rope::RopeConfigCommon,
        },
        *,
    };

    #[test]
    fn test_decoder_config() {
        let config_str = r#"
            {
                "embedding_config": {
                    "type": "QuantizedTiedEmbeddingConfig",
                    "input_scale": null,
                    "logit_soft_cap": null,
                    "embedding_quantization_mode": "int8",
                    "activation_quantization_mode": "int8",
                    "activation_precision": "bfloat16"
                },
                "global_rope_config": {
                    "type": "LlamaRoPEConfig",
                    "precision": "bfloat16",
                    "base": 500000.0,
                    "max_sequence_length": 262144,
                    "scaling_factor": 32.0,
                    "original_context_length": 8192,
                    "low_frequency_factor": 1.0,
                    "high_frequency_factor": 4.0
                },
                "local_rope_config": null,
                "layer_config": {
                    "pre_attention_norm_config": {
                        "scale_precision": "bfloat16",
                        "accumulation_precision": "float32",
                        "epsilon": 1e-05,
                        "scale_offset": null,
                        "upcast_mode": "only_normalization"
                    },
                    "mixer_config": {
                        "type": "AttentionConfig",
                        "qkv_projection_config": {
                            "type": "QLoRALinearConfig",
                            "group_size": 32,
                            "weight_quantization_mode": "uint4",
                            "activation_quantization_mode": "int8",
                            "activation_precision": "bfloat16",
                            "lora_rank": 16,
                            "lora_scale": 2.0
                        },
                        "out_projection_config": {
                            "type": "QLoRALinearConfig",
                            "group_size": 32,
                            "weight_quantization_mode": "uint4",
                            "activation_quantization_mode": "int8",
                            "activation_precision": "bfloat16",
                            "lora_rank": 16,
                            "lora_scale": 2.0
                        },
                        "query_norm_config": null,
                        "key_norm_config": null,
                        "num_heads": 12,
                        "num_groups": 12,
                        "head_dim": 64,
                        "is_causal": true,
                        "scale": null,
                        "sliding_window_size": null,
                        "logit_soft_cap": null,
                        "has_sinks": false,
                        "has_qkv_biases": false,
                        "has_out_biases": false
                    },
                    "post_attention_norm_config": null,
                    "pre_mlp_norm_config": {
                        "scale_precision": "bfloat16",
                        "accumulation_precision": "float32",
                        "epsilon": 1e-05,
                        "scale_offset": null,
                        "upcast_mode": "only_normalization"
                    },
                    "mlp_config": {
                        "type": "DenseMLPConfig",
                        "linear_config": {
                            "type": "QLoRALinearConfig",
                            "group_size": 32,
                            "weight_quantization_mode": "uint4",
                            "activation_quantization_mode": "int8",
                            "activation_precision": "bfloat16",
                            "lora_rank": 16,
                            "lora_scale": 2.0
                        },
                        "activation": {"type": "SiLU"}
                    },
                    "post_mlp_norm_config": null
                },
                "output_norm_config": {
                    "scale_precision": "bfloat16",
                    "accumulation_precision": "float32",
                    "epsilon": 1e-05,
                    "scale_offset": null,
                    "upcast_mode": "only_normalization"
                },
                "vocab_size": 128256,
                "model_dim": 2048,
                "hidden_dim": 8192,
                "num_heads": 32,
                "num_groups": 8,
                "head_dim": 64,
                "attention_scale": null,
                "num_layers": 16,
                "sliding_window_sizes": null,
                "context_length": 8192
            }
        "#;

        let ground_truth_config = DecoderConfig {
            embedding_config: EmbeddingConfig::QuantizedTied {
                common: EmbeddingConfigCommon {
                    input_scale: None,
                    logit_soft_cap: None,
                },
                embedding_quantization_mode: QuantizationMode::Int8,
                activation_quantization_mode: Some(QuantizationMode::Int8),
                activation_precision: ConfigDataType::BFloat16,
            },
            global_rope_config: Some(RoPEConfig::Llama {
                common: RopeConfigCommon {
                    precision: ConfigDataType::BFloat16,
                    base: 500000.0,
                    max_sequence_length: 262144,
                },
                scaling_factor: 32.0,
                original_context_length: 8192,
                low_frequency_factor: 1.0,
                high_frequency_factor: 4.0,
            }),
            local_rope_config: None,
            layer_config: DecoderLayerConfig {
                pre_attention_norm_config: NormalizationConfig {
                    scale_precision: ConfigDataType::BFloat16,
                    accumulation_precision: ConfigDataType::Float32,
                    epsilon: 1e-5,
                    scale_offset: None,
                    upcast_mode: UpcastMode::OnlyNormalization,
                    subtract_mean: false,
                },
                pre_mlp_norm_config: NormalizationConfig {
                    scale_precision: ConfigDataType::BFloat16,
                    accumulation_precision: ConfigDataType::Float32,
                    epsilon: 1e-5,
                    scale_offset: None,
                    upcast_mode: UpcastMode::OnlyNormalization,
                    subtract_mean: false,
                },
                mixer_config: MixerConfig::Attention(AttentionConfig {
                    qkv_projection_config: LinearConfig::QLoRA {
                        quantization: QuantizationConfig {
                            group_size: 32,
                            weight_quantization_mode: QuantizationMode::UInt4,
                            activation_quantization_mode: Some(
                                QuantizationMode::Int8,
                            ),
                            activation_precision: ConfigDataType::BFloat16,
                        },
                        lora_rank: 16,
                        lora_scale: 2.0,
                    },
                    out_projection_config: LinearConfig::QLoRA {
                        quantization: QuantizationConfig {
                            group_size: 32,
                            weight_quantization_mode: QuantizationMode::UInt4,
                            activation_quantization_mode: Some(
                                QuantizationMode::Int8,
                            ),
                            activation_precision: ConfigDataType::BFloat16,
                        },
                        lora_rank: 16,
                        lora_scale: 2.0,
                    },
                    query_norm_config: None,
                    key_norm_config: None,
                    num_heads: Some(12),
                    num_groups: Some(12),
                    head_dim: Some(64),
                    is_causal: Some(true),
                    scale: None,
                    sliding_window_size: None,
                    logit_soft_cap: None,
                    has_sinks: false,
                    has_qkv_biases: false,
                    has_out_biases: false,
                }),
                mlp_config: MLPConfig::Dense(DenseMLPConfig {
                    linear_config: LinearConfig::QLoRA {
                        quantization: QuantizationConfig {
                            group_size: 32,
                            weight_quantization_mode: QuantizationMode::UInt4,
                            activation_quantization_mode: Some(
                                QuantizationMode::Int8,
                            ),
                            activation_precision: ConfigDataType::BFloat16,
                        },
                        lora_rank: 16,
                        lora_scale: 2.0,
                    },
                    activation: Activation::SiLU {
                        alpha: 1.0,
                    },
                    has_up_biases: false,
                    has_down_biases: false,
                    gate_clipping: None,
                    up_clipping: None,
                    activation_to_gate: true,
                }),
                post_attention_norm_config: None,
                post_mlp_norm_config: None,
            },
            output_norm_config: NormalizationConfig {
                scale_precision: ConfigDataType::BFloat16,
                accumulation_precision: ConfigDataType::Float32,
                epsilon: 1e-5,
                scale_offset: None,
                upcast_mode: UpcastMode::OnlyNormalization,
                subtract_mean: false,
            },
            layer_configs: None,
            vocab_size: 128256,
            model_dim: 2048,
            hidden_dim: 8192,
            num_heads: 32,
            num_groups: 8,
            head_dim: 64,
            attention_scale: None,
            num_layers: 16,
            sliding_window_sizes: None,
            layer_types: None,
            context_length: 8192,
        };

        let deserialized_config: DecoderConfig = from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);
    }

    #[test]
    fn test_decoder_config_with_layer_configs() {
        let config_str = r#"
            {
                "embedding_config": {
                    "type": "TiedEmbeddingConfig",
                    "input_scale": null,
                    "logit_soft_cap": null,
                    "precision": "bfloat16"
                },
                "global_rope_config": null,
                "local_rope_config": null,
                "layer_configs": [
                    {
                        "pre_mixer_norm_config": {
                            "scale_precision": "bfloat16",
                            "accumulation_precision": "float32",
                            "epsilon": 1e-06,
                            "scale_offset": 1.0,
                            "upcast_mode": "full_layer"
                        },
                        "mixer_config": {
                            "type": "AttentionConfig",
                            "qkv_projection_config": {
                                "type": "FullPrecisionLinearConfig",
                                "precision": "bfloat16"
                            },
                            "out_projection_config": {
                                "type": "FullPrecisionLinearConfig",
                                "precision": "bfloat16"
                            },
                            "query_norm_config": null,
                            "key_norm_config": null,
                            "num_heads": 4,
                            "num_groups": 1,
                            "head_dim": 256,
                            "is_causal": true,
                            "scale": 0.0625,
                            "sliding_window_size": 512,
                            "logit_soft_cap": null,
                            "has_sinks": false,
                            "has_qkv_biases": false,
                            "has_out_biases": false
                        },
                        "post_mixer_norm_config": null,
                        "pre_mlp_norm_config": {
                            "scale_precision": "bfloat16",
                            "accumulation_precision": "float32",
                            "epsilon": 1e-06,
                            "scale_offset": 1.0,
                            "upcast_mode": "full_layer"
                        },
                        "mlp_config": {
                            "type": "DenseMLPConfig",
                            "linear_config": {
                                "type": "FullPrecisionLinearConfig",
                                "precision": "bfloat16"
                            },
                            "activation": {
                                "type": "GELU"
                            }
                        },
                        "post_mlp_norm_config": null
                    },
                    {
                        "pre_mixer_norm_config": {
                            "scale_precision": "bfloat16",
                            "accumulation_precision": "float32",
                            "epsilon": 1e-06,
                            "scale_offset": 1.0,
                            "upcast_mode": "full_layer"
                        },
                        "mixer_config": {
                            "type": "AttentionConfig",
                            "qkv_projection_config": {
                                "type": "FullPrecisionLinearConfig",
                                "precision": "bfloat16"
                            },
                            "out_projection_config": {
                                "type": "FullPrecisionLinearConfig",
                                "precision": "bfloat16"
                            },
                            "query_norm_config": null,
                            "key_norm_config": null,
                            "num_heads": 4,
                            "num_groups": 1,
                            "head_dim": 256,
                            "is_causal": true,
                            "scale": 0.0625,
                            "sliding_window_size": 256,
                            "logit_soft_cap": null,
                            "has_sinks": false,
                            "has_qkv_biases": false,
                            "has_out_biases": false
                        },
                        "post_mixer_norm_config": null,
                        "pre_mlp_norm_config": {
                            "scale_precision": "bfloat16",
                            "accumulation_precision": "float32",
                            "epsilon": 1e-06,
                            "scale_offset": 1.0,
                            "upcast_mode": "full_layer"
                        },
                        "mlp_config": {
                            "type": "DenseMLPConfig",
                            "linear_config": {
                                "type": "FullPrecisionLinearConfig",
                                "precision": "bfloat16"
                            },
                            "activation": {
                                "type": "GELU"
                            }
                        },
                        "post_mlp_norm_config": null
                    }
                ],
                "output_norm_config": {
                    "scale_precision": "bfloat16",
                    "accumulation_precision": "float32",
                    "epsilon": 1e-06,
                    "scale_offset": 1.0,
                    "upcast_mode": "full_layer"
                },
                "vocab_size": 32000,
                "model_dim": 1152,
                "hidden_dim": 6912,
                "context_length": 32768
            }
        "#;

        let config: DecoderConfig = from_str(config_str).unwrap();
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.num_groups, 1);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.attention_scale, Some(0.0625));
        let sliding_windows = config.sliding_window_sizes.unwrap();
        assert_eq!(sliding_windows.len(), 2);
        assert_eq!(sliding_windows[0], Some(512));
        assert_eq!(sliding_windows[1], Some(256));
        let layer_types = config.layer_types.unwrap();
        assert!(matches!(layer_types[0], DecoderLayerType::Transformer));
        assert_eq!(config.layer_configs.unwrap().len(), 2);
    }
}
