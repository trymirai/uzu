use serde::{Deserialize, Serialize};

use super::{
    attention::AttentionConfig, mamba::Mamba2Config, mlp::MLPConfig,
    normalization::RMSNormConfig,
};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum MixerConfig {
    #[serde(rename = "AttentionConfig")]
    Attention(AttentionConfig),
    #[serde(rename = "Mamba2Config")]
    Mamba(Mamba2Config),
}

impl MixerConfig {
    pub fn as_attention(&self) -> Option<&AttentionConfig> {
        match self {
            MixerConfig::Attention(config) => Some(config),
            _ => None,
        }
    }

    pub fn as_mamba(&self) -> Option<&Mamba2Config> {
        match self {
            MixerConfig::Mamba(config) => Some(config),
            _ => None,
        }
    }

    pub fn num_heads(&self) -> Option<usize> {
        match self {
            MixerConfig::Attention(config) => config.num_heads,
            MixerConfig::Mamba(config) => Some(config.num_heads),
        }
    }

    pub fn num_groups(&self) -> Option<usize> {
        match self {
            MixerConfig::Attention(config) => config.num_groups,
            MixerConfig::Mamba(config) => Some(config.num_groups),
        }
    }

    pub fn head_dim(&self) -> Option<usize> {
        match self {
            MixerConfig::Attention(config) => config.head_dim,
            MixerConfig::Mamba(config) => Some(config.head_dim),
        }
    }

    pub fn sliding_window_size(&self) -> Option<usize> {
        match self {
            MixerConfig::Attention(config) => config.sliding_window_size,
            MixerConfig::Mamba(_) => None,
        }
    }

    pub fn attention_scale(&self) -> Option<f32> {
        match self {
            MixerConfig::Attention(config) => config.scale,
            MixerConfig::Mamba(_) => None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DecoderLayerConfig {
    #[serde(alias = "pre_mixer_norm_config")]
    pub pre_attention_norm_config: RMSNormConfig,
    #[serde(alias = "mixer_config")]
    pub mixer_config: MixerConfig,
    #[serde(alias = "post_mixer_norm_config")]
    pub post_attention_norm_config: Option<RMSNormConfig>,
    #[serde(alias = "pre_mlp_norm_config")]
    pub pre_mlp_norm_config: RMSNormConfig,
    pub mlp_config: MLPConfig,
    #[serde(alias = "post_mlp_norm_config")]
    pub post_mlp_norm_config: Option<RMSNormConfig>,
}

impl DecoderLayerConfig {
    pub fn attention_config(&self) -> Option<&AttentionConfig> {
        self.mixer_config.as_attention()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::from_str;

    use super::{
        super::{
            common::{Activation, ConfigDataType, QuantizationMode},
            linear::{LinearConfig, QuantizationConfig},
            normalization::UpcastMode,
        },
        *,
    };
    use crate::config::mlp;

    #[test]
    fn test_decoder_layer_config() {
        let config_str = r#"
            {
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
            }
        "#;

        let ground_truth_config = DecoderLayerConfig {
            pre_attention_norm_config: RMSNormConfig {
                scale_precision: ConfigDataType::BFloat16,
                accumulation_precision: ConfigDataType::Float32,
                epsilon: 1e-5,
                scale_offset: None,
                upcast_mode: UpcastMode::OnlyNormalization,
            },
            pre_mlp_norm_config: RMSNormConfig {
                scale_precision: ConfigDataType::BFloat16,
                accumulation_precision: ConfigDataType::Float32,
                epsilon: 1e-5,
                scale_offset: None,
                upcast_mode: UpcastMode::OnlyNormalization,
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
                logit_soft_cap: None,
                has_sinks: false,
                has_qkv_biases: false,
                has_out_biases: false,
                num_heads: None,
                num_groups: None,
                head_dim: None,
                is_causal: None,
                scale: None,
                sliding_window_size: None,
            }),
            mlp_config: MLPConfig::Dense(mlp::DenseMLPConfig {
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
            }),
            post_attention_norm_config: None,
            post_mlp_norm_config: None,
        };

        let deserialized_config: DecoderLayerConfig =
            from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);
    }
}
