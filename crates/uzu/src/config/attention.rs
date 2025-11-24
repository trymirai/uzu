use serde::{Deserialize, Serialize};

use super::{RMSNormConfig, linear::LinearConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct AttentionConfig {
    pub qkv_projection_config: LinearConfig,
    pub out_projection_config: LinearConfig,

    pub query_norm_config: Option<RMSNormConfig>,
    pub key_norm_config: Option<RMSNormConfig>,

    pub num_heads: Option<usize>,
    pub num_groups: Option<usize>,
    pub head_dim: Option<usize>,
    pub is_causal: Option<bool>,
    pub scale: Option<f32>,
    pub sliding_window_size: Option<usize>,

    pub logit_soft_cap: Option<f32>,
    pub has_sinks: bool,
    pub has_qkv_biases: bool,
    pub has_out_biases: bool,
}

#[cfg(test)]
mod tests {
    use serde_json5::from_str;

    use super::{
        super::{
            common::{ConfigDataType, QuantizationMode},
            linear::QuantizationConfig,
        },
        *,
    };

    #[test]
    fn test_attention_config() {
        let config_str = r#"
            {
                "qkv_projection_config": {
                    "type": "QLoRALinearConfig",
                    "group_size": 32,
                    "weight_quantization_mode": "uint4",
                    "activation_quantization_mode": "int8",
                    "activation_precision": "bfloat16",
                    "lora_rank": 16,
                    "lora_scale": 2.0,
                },
                "out_projection_config": {
                    "type": "QLoRALinearConfig",
                    "group_size": 32,
                    "weight_quantization_mode": "uint4",
                    "activation_quantization_mode": "int8",
                    "activation_precision": "bfloat16",
                    "lora_rank": 16,
                    "lora_scale": 2.0,
                },
                "query_norm_config": null,
                "key_norm_config": null,
                "logit_soft_cap": null,
                "has_sinks": false,
                "has_qkv_biases": false,
                "has_out_biases": false
            }
        "#;

        let ground_truth_config = AttentionConfig {
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
            num_heads: None,
            num_groups: None,
            head_dim: None,
            is_causal: None,
            scale: None,
            sliding_window_size: None,
            logit_soft_cap: None,
            has_sinks: false,
            has_qkv_biases: false,
            has_out_biases: false,
        };

        let deserialized_config: AttentionConfig =
            from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);
    }
}
