use serde::{Deserialize, Serialize};

use crate::{ConfigDataType, backends::common::gpu_types::QuantizationMode};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct QuantizationConfig {
    pub group_size: usize,
    pub weight_quantization_mode: QuantizationMode,
    pub activation_quantization_mode: Option<QuantizationMode>,
    pub activation_precision: ConfigDataType,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum LinearConfig {
    #[serde(rename = "FullPrecisionLinearConfig")]
    FullPrecision {
        precision: ConfigDataType,
    },
    #[serde(rename = "GroupQuantizedLinearConfig")]
    Quantized(QuantizationConfig),
    #[serde(rename = "MLXQuantizedLinearConfig")]
    MLXQuantized(QuantizationConfig),
    #[serde(rename = "QLoRALinearConfig")]
    QLoRA {
        #[serde(flatten)]
        quantization: QuantizationConfig,
        lora_rank: usize,
        lora_scale: f32,
    },
    #[serde(rename = "RHTLinearWrapperConfig")]
    RHTLinearWrapper {
        block_size: usize,
        inner_config: Box<LinearConfig>,
    },
}

impl LinearConfig {
    pub fn activation_precision(&self) -> ConfigDataType {
        match self {
            LinearConfig::FullPrecision {
                precision,
            } => *precision,
            LinearConfig::Quantized(quantization) => quantization.activation_precision,
            LinearConfig::MLXQuantized(quantization) => quantization.activation_precision,
            LinearConfig::QLoRA {
                quantization,
                ..
            } => quantization.activation_precision,
            LinearConfig::RHTLinearWrapper {
                inner_config,
                ..
            } => inner_config.activation_precision(),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json5::from_str;

    use super::*;

    #[test]
    fn test_linear_config() {
        let config_str = r#"
            {
                "type": "QLoRALinearConfig",
                "group_size": 32,
                "weight_quantization_mode": "uint4",
                "activation_quantization_mode": "int8",
                "activation_precision": "bfloat16",
                "lora_rank": 16,
                "lora_scale": 2.0,
            }
        "#;

        let ground_truth_config = LinearConfig::QLoRA {
            quantization: QuantizationConfig {
                group_size: 32,
                weight_quantization_mode: QuantizationMode::UINT4,
                activation_quantization_mode: Some(QuantizationMode::INT8),
                activation_precision: ConfigDataType::BFloat16,
            },
            lora_rank: 16,
            lora_scale: 2.0,
        };

        let deserialized_config: LinearConfig = from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);
    }

    #[test]
    fn test_rht_linear_wrapper_config() {
        let config_str = r#"
            {
                "type": "RHTLinearWrapperConfig",
                "block_size": 32,
                "inner_config": {
                    "type": "GroupQuantizedLinearConfig",
                    "group_size": 32,
                    "weight_quantization_mode": "uint4",
                    "activation_quantization_mode": null,
                    "activation_precision": "bfloat16"
                }
            }
        "#;

        let ground_truth_config = LinearConfig::RHTLinearWrapper {
            block_size: 32,
            inner_config: Box::new(LinearConfig::Quantized(QuantizationConfig {
                group_size: 32,
                weight_quantization_mode: QuantizationMode::UINT4,
                activation_quantization_mode: None,
                activation_precision: ConfigDataType::BFloat16,
            })),
        };

        let deserialized_config: LinearConfig = from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);
    }
}
