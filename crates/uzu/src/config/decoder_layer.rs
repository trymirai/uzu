use serde::{Deserialize, Serialize};

use super::{
    attention::AttentionConfig, mlp::MLPConfig, normalization::RMSNormConfig,
};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DecoderLayerConfig {
    pub pre_attention_norm_config: RMSNormConfig,
    pub attention_config: AttentionConfig,
    pub post_attention_norm_config: Option<RMSNormConfig>,
    pub pre_mlp_norm_config: RMSNormConfig,
    pub mlp_config: MLPConfig,
    pub post_mlp_norm_config: Option<RMSNormConfig>,
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
                "attention_config": {
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
            attention_config: AttentionConfig {
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
            },
            mlp_config: MLPConfig {
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
                activation: Activation::SILU,
            },
            post_attention_norm_config: None,
            post_mlp_norm_config: None,
        };

        let deserialized_config: DecoderLayerConfig =
            from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);
    }
}
