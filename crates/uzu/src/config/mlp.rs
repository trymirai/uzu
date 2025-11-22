use serde::{Deserialize, Serialize};

use super::{common::Activation, linear::LinearConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum MLPConfig {
    #[serde(rename = "DenseMLPConfig")]
    Dense(DenseMLPConfig),
    #[serde(rename = "MixtureOfExpertsConfig")]
    MixtureOfExperts(MixtureOfExpertsConfig),
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DenseMLPConfig {
    pub linear_config: LinearConfig,
    pub activation: Activation,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct MixtureOfExpertsConfig {
    pub mixture_size: usize,
    pub num_experts_per_token: usize,
    pub routing_function: RoutingFunctionConfig,
    pub router_config: LinearConfig,
    pub router_has_biases: bool,
    pub expert_config: MoeExpertConfig,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum RoutingFunctionConfig {
    #[serde(rename = "SoftmaxRouting")]
    SoftmaxRouting,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct MoeExpertConfig {
    pub linear_config: LinearConfig,
    pub activation: Activation,
    pub has_up_biases: bool,
    pub has_down_biases: bool,
    pub gate_clipping: [Option<f32>; 2],
    pub up_clipping: [f32; 2],
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

        let ground_truth_config = MLPConfig::Dense(DenseMLPConfig {
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
        });

        let deserialized_config: MLPConfig = from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);
    }

    #[test]
    fn test_moe_mlp_config() {
        let config_str = r#"
            {
                "type": "MixtureOfExpertsConfig",
                "mixture_size": 32,
                "num_experts_per_token": 4,
                "routing_function": {"type": "SoftmaxRouting"},
                "router_config": {"type": "FullPrecisionLinearConfig", "precision": "bfloat16"},
                "router_has_biases": true,
                "expert_config": {
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
                    "has_up_biases": true,
                    "has_down_biases": true,
                    "gate_clipping": [null, 7.0],
                    "up_clipping": [-6.0, 8.0]
                }
            }
        "#;

        let ground_truth_config =
            MLPConfig::MixtureOfExperts(MixtureOfExpertsConfig {
                mixture_size: 32,
                num_experts_per_token: 4,
                routing_function: RoutingFunctionConfig::SoftmaxRouting,
                router_config: LinearConfig::FullPrecision {
                    precision: ConfigDataType::BFloat16,
                },
                router_has_biases: true,
                expert_config: MoeExpertConfig {
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
                    has_up_biases: true,
                    has_down_biases: true,
                    gate_clipping: [None, Some(7.0)],
                    up_clipping: [-6.0, 8.0],
                },
            });

        let deserialized_config: MLPConfig = from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);
    }
}
