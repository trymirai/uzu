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
