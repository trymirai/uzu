use serde_json::from_str;

use super::*;

#[test]
fn test_rope_config() {
    let config_str = r#"
            {
                "type": "LlamaRoPEConfig",
                "precision": "bfloat16",
                "base": 500000.0,
                "max_sequence_length": 131072,
                "scaling_factor": 32.0,
                "original_context_length": 8192,
                "low_frequency_factor": 1.0,
                "high_frequency_factor": 4.0
            }
        "#;

    let ground_truth_config = RoPEConfig::Llama {
        common: RopeConfigCommon {
            precision: DataType::BF16,
            base: 500000.0,
            max_sequence_length: 131072,
        },
        scaling_factor: 32.0,
        original_context_length: 8192,
        low_frequency_factor: 1.0,
        high_frequency_factor: 4.0,
    };

    let deserialized_config: RoPEConfig = from_str(config_str).unwrap();
    assert_eq!(deserialized_config, ground_truth_config);
}
