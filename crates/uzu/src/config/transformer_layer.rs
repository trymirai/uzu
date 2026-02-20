use serde::{Deserialize, Serialize};

use crate::{AttentionConfig, MLPConfig, MixerConfig, NormalizationConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct TransformerLayerConfig {
    #[serde(alias = "pre_mixer_norm_config")]
    pub pre_attention_norm_config: Option<NormalizationConfig>,
    pub mixer_config: MixerConfig,
    #[serde(alias = "post_mixer_norm_config")]
    pub post_attention_norm_config: Option<NormalizationConfig>,
    pub pre_mlp_norm_config: NormalizationConfig,
    pub mlp_config: MLPConfig,
    pub post_mlp_norm_config: Option<NormalizationConfig>,
}

impl TransformerLayerConfig {
    pub fn attention_config(&self) -> Option<&AttentionConfig> {
        self.mixer_config.as_attention()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::from_str;

    use super::{
        super::{
            attention::AttentionConfig,
            common::{Activation, ConfigDataType, QuantizationMode},
            linear::{LinearConfig, QuantizationConfig},
            normalization::UpcastMode,
        },
        *,
    };
    use crate::config::mlp;

    #[test]
    fn test_transformer_layer_config() {
        let config_str = r#"
            {
                "pre_attention_norm_config": {
                    "scale_precision": "bfloat16",
                    "accumulation_precision": "float32",
                    "epsilon": 1e-05,
                    "scale_offset": null,
                    "upcast_mode": "only_normalization",
                    "subtract_mean": false
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
                    "is_causal": false,
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
                    "upcast_mode": "only_normalization",
                    "subtract_mean": false
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
                    "activation": {"type": "SiLU"},
                    "has_up_biases": false,
                    "has_down_biases": false,
                    "gate_clipping": null,
                    "up_clipping": null,
                    "activation_to_gate": false
                },
                "post_mlp_norm_config": null
            }
        "#;

        let ground_truth_config = TransformerLayerConfig {
            pre_attention_norm_config: Some(NormalizationConfig {
                scale_precision: ConfigDataType::BFloat16,
                accumulation_precision: ConfigDataType::Float32,
                epsilon: 1e-5,
                scale_offset: None,
                upcast_mode: UpcastMode::OnlyNormalization,
                subtract_mean: false,
            }),
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
                        activation_quantization_mode: Some(QuantizationMode::Int8),
                        activation_precision: ConfigDataType::BFloat16,
                    },
                    lora_rank: 16,
                    lora_scale: 2.0,
                },
                out_projection_config: LinearConfig::QLoRA {
                    quantization: QuantizationConfig {
                        group_size: 32,
                        weight_quantization_mode: QuantizationMode::UInt4,
                        activation_quantization_mode: Some(QuantizationMode::Int8),
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
                is_causal: Some(false),
                scale: None,
                sliding_window_size: None,
                logit_soft_cap: None,
                has_sinks: false,
                has_qkv_biases: false,
                has_out_biases: false,
            }),
            mlp_config: MLPConfig::Dense(mlp::DenseMLPConfig {
                linear_config: LinearConfig::QLoRA {
                    quantization: QuantizationConfig {
                        group_size: 32,
                        weight_quantization_mode: QuantizationMode::UInt4,
                        activation_quantization_mode: Some(QuantizationMode::Int8),
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
                activation_to_gate: false,
            }),
            post_attention_norm_config: None,
            post_mlp_norm_config: None,
        };

        let deserialized_config: TransformerLayerConfig = from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);
    }
}
