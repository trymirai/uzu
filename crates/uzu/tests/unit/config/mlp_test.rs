use serde_json::from_str;

use super::{super::linear::QuantizationConfig, *};
use crate::{backends::common::gpu_types::QuantizationMode, config::ConfigDataType};

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
                weight_quantization_mode: QuantizationMode::UINT4,
                activation_quantization_mode: Some(QuantizationMode::INT8),
                activation_precision: ConfigDataType::BFloat16,
            },
            lora_rank: 16,
            lora_scale: 2.0,
        },
        activation: ActivationConfig::silu_default(),
        has_up_biases: false,
        has_down_biases: false,
        gate_clipping: None,
        up_clipping: None,
        activation_to_gate: true,
    });

    let deserialized_config: MLPConfig = from_str(config_str).unwrap();
    assert_eq!(deserialized_config, ground_truth_config);
}

#[test]
fn test_moe_mlp_config() {
    let config_str = r#"
            {
                "type": "MixtureOfExpertsConfig",
                "num_routed_experts": 32,
                "num_active_routed_experts": 4,
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

    let ground_truth_config = MLPConfig::MixtureOfExperts(MixtureOfExpertsConfig {
        num_routed_experts: 32,
        num_active_routed_experts: 4,
        routing_function: RoutingFunctionConfig::SoftmaxRouting,
        router_config: LinearConfig::FullPrecision {
            precision: ConfigDataType::BFloat16,
        },
        router_has_biases: true,
        expert_config: MoeExpertConfig {
            linear_config: LinearConfig::QLoRA {
                quantization: QuantizationConfig {
                    group_size: 32,
                    weight_quantization_mode: QuantizationMode::UINT4,
                    activation_quantization_mode: Some(QuantizationMode::INT8),
                    activation_precision: ConfigDataType::BFloat16,
                },
                lora_rank: 16,
                lora_scale: 2.0,
            },
            activation: ActivationConfig::silu_default(),
            has_up_biases: true,
            has_down_biases: true,
            gate_clipping: [None, Some(7.0)],
            up_clipping: [-6.0, 8.0],
        },
    });

    let deserialized_config: MLPConfig = from_str(config_str).unwrap();
    assert_eq!(deserialized_config, ground_truth_config);
}

#[test]
fn test_moe_mlp_config_with_additional_fields() {
    let config_str = r#"
            {
                "type": "MixtureOfExpertsConfig",
                "num_routed_experts": 32,
                "num_active_routed_experts": 4,
                "routing_function": {"type": "SoftmaxRouting"},
                "router_config": {"type": "FullPrecisionLinearConfig", "precision": "bfloat16"},
                "router_has_biases": true,
                "num_shared_experts": 0,
                "expert_hidden_dim": 2880,
                "gate_config": null,
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

    let ground_truth_config = MLPConfig::MixtureOfExperts(MixtureOfExpertsConfig {
        num_routed_experts: 32,
        num_active_routed_experts: 4,
        routing_function: RoutingFunctionConfig::SoftmaxRouting,
        router_config: LinearConfig::FullPrecision {
            precision: ConfigDataType::BFloat16,
        },
        router_has_biases: true,
        expert_config: MoeExpertConfig {
            linear_config: LinearConfig::QLoRA {
                quantization: QuantizationConfig {
                    group_size: 32,
                    weight_quantization_mode: QuantizationMode::UINT4,
                    activation_quantization_mode: Some(QuantizationMode::INT8),
                    activation_precision: ConfigDataType::BFloat16,
                },
                lora_rank: 16,
                lora_scale: 2.0,
            },
            activation: ActivationConfig::silu_default(),
            has_up_biases: true,
            has_down_biases: true,
            gate_clipping: [None, Some(7.0)],
            up_clipping: [-6.0, 8.0],
        },
    });

    let deserialized_config: MLPConfig = from_str(config_str).unwrap();
    assert_eq!(deserialized_config, ground_truth_config);
}
