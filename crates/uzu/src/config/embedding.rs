use serde::{Deserialize, Serialize};

use super::common::{ConfigDataType, QuantizationMode};

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

        let deserialized_config: EmbeddingConfig =
            from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);
    }
}
