use serde_json::from_str;

use super::*;

#[test]
fn test_normalization_config() {
    let config_str = r#"
            {
                "scale_precision": "bfloat16",
                "accumulation_precision": "float32",
                "epsilon": 1e-05,
                "scale_offset": null,
                "upcast_mode": "only_normalization"
            }
        "#;

    let ground_truth_config = NormalizationConfig {
        scale_precision: ConfigDataType::BFloat16,
        accumulation_precision: ConfigDataType::Float32,
        epsilon: 1e-05,
        scale_offset: None,
        upcast_mode: UpcastMode::OnlyNormalization,
        subtract_mean: false,
        use_bias: false,
        has_scale: true,
    };

    let deserialized_config: NormalizationConfig = from_str(config_str).unwrap();
    assert_eq!(deserialized_config, ground_truth_config);
}
