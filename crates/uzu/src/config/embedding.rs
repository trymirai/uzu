use serde::{Deserialize, Serialize};

use crate::{ConfigDataType, backends::common::gpu_types::QuantizationMode};

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
#[path = "../../tests_unit/config/embedding_test.rs"]
mod tests;
