use serde::{Deserialize, Serialize};

use super::{common::Activation, linear::LinearConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct MLPConfig {
    pub linear_config: LinearConfig,
    pub activation: Activation,
}

#[cfg(test)]
mod tests {
    use serde_json::from_str;

    use super::{
        super::{
            common::{ConfigDataType, QuantizationMode},
            linear::QuantizationConfig,
        },
        *,
    };

    #[test]
    fn test_mlp_config() {
        let config_str = r#"
            {
                "linear_config": {
                    "type": "QLoRALinearConfig",
                    "group_size": 32,
                    "weight_quantization_mode": "int4",
                    "activation_quantization_mode": "int8",
                    "activation_precision": "bfloat16",
                    "lora_rank": 16,
                    "lora_scale": 2.0
                },
                "activation": "silu"
            }
        "#;

        let ground_truth_config = MLPConfig {
            linear_config: LinearConfig::QLoRA {
                quantization: QuantizationConfig {
                    group_size: 32,
                    weight_quantization_mode: QuantizationMode::Int4,
                    activation_quantization_mode: Some(QuantizationMode::Int8),
                    activation_precision: ConfigDataType::BFloat16,
                },
                lora_rank: 16,
                lora_scale: 2.0,
            },
            activation: Activation::SILU,
        };

        let deserialized_config: MLPConfig = from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);
    }
}
