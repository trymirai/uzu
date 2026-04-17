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
    #[serde(rename = "MLXQuantizedOutputUntiedEmbeddingConfig")]
    MLXQuantizedOutputUntied {
        #[serde(flatten)]
        common: EmbeddingConfigCommon,
        group_size: usize,
        output_group_size: usize,
        embedding_quantization_mode: QuantizationMode,
        activation_precision: ConfigDataType,
        output_quantization_mode: QuantizationMode,
    },
}

impl EmbeddingConfig {
    pub fn common(&self) -> &EmbeddingConfigCommon {
        match self {
            EmbeddingConfig::Tied {
                common,
                ..
            }
            | EmbeddingConfig::Untied {
                common,
                ..
            }
            | EmbeddingConfig::MLXQuantizedTied {
                common,
                ..
            }
            | EmbeddingConfig::MLXQuantizedUntied {
                common,
                ..
            }
            | EmbeddingConfig::MLXSemiQuantizedUntied {
                common,
                ..
            }
            | EmbeddingConfig::MLXQuantizedOutputUntied {
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

        let mlx_quant_output_untied_str = r#"
            {
                "type": "MLXQuantizedOutputUntiedEmbeddingConfig",
                "input_scale": null,
                "logit_soft_cap": null,
                "group_size": 128,
                "output_group_size": 512,
                "embedding_quantization_mode": "uint8",
                "activation_precision": "bfloat16",
                "output_quantization_mode": "uint2"
            }
        "#;

        let mlx_quant_output_untied = EmbeddingConfig::MLXQuantizedOutputUntied {
            common: EmbeddingConfigCommon {
                input_scale: None,
                logit_soft_cap: None,
            },
            group_size: 128,
            output_group_size: 512,
            embedding_quantization_mode: QuantizationMode::UInt8,
            activation_precision: ConfigDataType::BFloat16,
            output_quantization_mode: QuantizationMode::UInt2,
        };

        let deserialized: EmbeddingConfig = from_str(mlx_quant_output_untied_str).unwrap();
        assert_eq!(deserialized, mlx_quant_output_untied);
    }
}
