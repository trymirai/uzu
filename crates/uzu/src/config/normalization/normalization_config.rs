use serde::{Deserialize, Serialize};

use super::UpcastMode;
use crate::ConfigDataType;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct NormalizationConfig {
    pub scale_precision: ConfigDataType,
    pub accumulation_precision: ConfigDataType,
    pub epsilon: f32,
    pub scale_offset: Option<f32>,
    pub upcast_mode: UpcastMode,
    #[serde(default)]
    pub subtract_mean: bool,
}

#[cfg(test)]
mod tests {
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
        };

        let deserialized_config: NormalizationConfig = from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);
    }
}
