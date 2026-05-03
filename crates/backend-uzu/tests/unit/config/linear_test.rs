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
            weight_quantization_mode: QuantizationMode::U4,
            activation_quantization_mode: Some(QuantizationMode::I8),
            activation_precision: DataType::BF16,
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
            weight_quantization_mode: QuantizationMode::U4,
            activation_quantization_mode: None,
            activation_precision: DataType::BF16,
        })),
    };

    let deserialized_config: LinearConfig = from_str(config_str).unwrap();
    assert_eq!(deserialized_config, ground_truth_config);
}
