use serde_json::from_str;

use super::*;

#[test]
fn test_embedding_config() {
    let semi_config_str = r#"
            {
                "type": "MLXSemiQuantizedUntiedEmbeddingConfig",
                "input_scale": null,
                "logit_soft_cap": null,
                "group_size": 128,
                "embedding_quantization_mode": "uint4",
                "activation_quantization_mode": null,
                "activation_precision": "bfloat16"
            }
        "#;

    let semi_config = EmbeddingConfig::MLXSemiQuantizedUntied {
        common: EmbeddingConfigCommon {
            input_scale: None,
            logit_soft_cap: None,
        },
        group_size: 128,
        embedding_quantization_mode: QuantizationMode::UINT4,
        activation_quantization_mode: None,
        activation_precision: ConfigDataType::BFloat16,
    };

    let deserialized: EmbeddingConfig = from_str(semi_config_str).unwrap();
    assert_eq!(deserialized, semi_config);

    let mlx_quant_untied_str = r#"
            {
                "type": "MLXQuantizedUntiedEmbeddingConfig",
                "input_scale": null,
                "logit_soft_cap": null,
                "group_size": 128,
                "embedding_quantization_mode": "uint4",
                "activation_quantization_mode": null,
                "activation_precision": "bfloat16"
            }
        "#;

    let mlx_quant_untied = EmbeddingConfig::MLXQuantizedUntied {
        common: EmbeddingConfigCommon {
            input_scale: None,
            logit_soft_cap: None,
        },
        group_size: 128,
        embedding_quantization_mode: QuantizationMode::UINT4,
        activation_quantization_mode: None,
        activation_precision: ConfigDataType::BFloat16,
    };

    let deserialized: EmbeddingConfig = from_str(mlx_quant_untied_str).unwrap();
    assert_eq!(deserialized, mlx_quant_untied);
}
