use serde::{Deserialize, Serialize};

use crate::{ConfigDataType, QuantizationMode};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct EmbeddingConfigCommon {
    pub input_scale: Option<f32>,
    pub logit_soft_cap: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum EmbeddingConfig {
    #[serde(rename = "TiedEmbeddingConfig")]
    Tied {
        #[serde(flatten)]
        common: EmbeddingConfigCommon,
        precision: ConfigDataType,
    },
    #[serde(rename = "UntiedEmbeddingConfig")]
    Untied {
        #[serde(flatten)]
        common: EmbeddingConfigCommon,
        precision: ConfigDataType,
    },
    #[serde(rename = "QuantizedTiedEmbeddingConfig")]
    QuantizedTied {
        #[serde(flatten)]
        common: EmbeddingConfigCommon,
        embedding_quantization_mode: QuantizationMode,
        activation_quantization_mode: Option<QuantizationMode>,
        activation_precision: ConfigDataType,
    },
    #[serde(rename = "MLXQuantizedTiedEmbeddingConfig")]
    MLXQuantizedTied {
        #[serde(flatten)]
        common: EmbeddingConfigCommon,
        group_size: usize,
        embedding_quantization_mode: QuantizationMode,
        activation_quantization_mode: Option<QuantizationMode>,
        activation_precision: ConfigDataType,
    },
    #[serde(rename = "MLXQuantizedUntiedEmbeddingConfig")]
    MLXQuantizedUntied {
        #[serde(flatten)]
        common: EmbeddingConfigCommon,
        group_size: usize,
        embedding_quantization_mode: QuantizationMode,
        activation_quantization_mode: Option<QuantizationMode>,
        activation_precision: ConfigDataType,
    },
    #[serde(rename = "MLXSemiQuantizedUntiedEmbeddingConfig")]
    MLXSemiQuantizedUntied {
        #[serde(flatten)]
        common: EmbeddingConfigCommon,
        group_size: usize,
        embedding_quantization_mode: QuantizationMode,
        activation_quantization_mode: Option<QuantizationMode>,
        activation_precision: ConfigDataType,
    },
}

impl EmbeddingConfig {
    pub fn common(&self) -> &EmbeddingConfigCommon {
        match self {
            EmbeddingConfig::Tied {
                common,
                ..
            } => common,
            EmbeddingConfig::Untied {
                common,
                ..
            } => common,
            EmbeddingConfig::QuantizedTied {
                common,
                ..
            } => common,
            EmbeddingConfig::MLXQuantizedTied {
                common,
                ..
            } => common,
            EmbeddingConfig::MLXQuantizedUntied {
                common,
                ..
            } => common,
            EmbeddingConfig::MLXSemiQuantizedUntied {
                common,
                ..
            } => common,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::from_str;

    use super::*;

    #[test]
    fn test_embedding_config() {
        let config_str = r#"
            {
                "type": "QuantizedTiedEmbeddingConfig",
                "input_scale": null,
                "logit_soft_cap": null,
                "embedding_quantization_mode": "int8",
                "activation_quantization_mode": "int8",
                "activation_precision": "bfloat16"
            }
        "#;

        let ground_truth_config = EmbeddingConfig::QuantizedTied {
            common: EmbeddingConfigCommon {
                input_scale: None,
                logit_soft_cap: None,
            },
            embedding_quantization_mode: QuantizationMode::Int8,
            activation_quantization_mode: Some(QuantizationMode::Int8),
            activation_precision: ConfigDataType::BFloat16,
        };

        let deserialized_config: EmbeddingConfig = from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);

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
            embedding_quantization_mode: QuantizationMode::UInt4,
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
            embedding_quantization_mode: QuantizationMode::UInt4,
            activation_quantization_mode: None,
            activation_precision: ConfigDataType::BFloat16,
        };

        let deserialized: EmbeddingConfig = from_str(mlx_quant_untied_str).unwrap();
        assert_eq!(deserialized, mlx_quant_untied);
    }
}
