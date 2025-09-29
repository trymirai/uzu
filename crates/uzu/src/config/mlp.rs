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

    use super::{super::linear::QuantizationConfig, *};
    use crate::config::{Activation, ConfigDataType, QuantizationMode};

    #[test]
    fn test_dense_mlp_config() {
        let config_str = r#"
            {
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
            }
        "#;

        let ground_truth_config = MLPConfig {
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
            activation: Activation::SILU,
        };

        let deserialized_config: MLPConfig = from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);
    }
}
