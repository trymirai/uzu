use serde::{Deserialize, Serialize};

use crate::{DataType, backends::common::gpu_types::QuantizationMode};

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
        precision: DataType,
    },
    #[serde(rename = "UntiedEmbeddingConfig")]
    Untied {
        #[serde(flatten)]
        common: EmbeddingConfigCommon,
        precision: DataType,
    },
    #[serde(rename = "MLXQuantizedTiedEmbeddingConfig")]
    MLXQuantizedTied {
        #[serde(flatten)]
        common: EmbeddingConfigCommon,
        group_size: usize,
        embedding_quantization_mode: QuantizationMode,
        activation_quantization_mode: Option<QuantizationMode>,
        activation_precision: DataType,
    },
    #[serde(rename = "MLXQuantizedUntiedEmbeddingConfig")]
    MLXQuantizedUntied {
        #[serde(flatten)]
        common: EmbeddingConfigCommon,
        group_size: usize,
        embedding_quantization_mode: QuantizationMode,
        activation_quantization_mode: Option<QuantizationMode>,
        activation_precision: DataType,
    },
    #[serde(rename = "MLXSemiQuantizedUntiedEmbeddingConfig")]
    MLXSemiQuantizedUntied {
        #[serde(flatten)]
        common: EmbeddingConfigCommon,
        group_size: usize,
        embedding_quantization_mode: QuantizationMode,
        activation_quantization_mode: Option<QuantizationMode>,
        activation_precision: DataType,
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
            } => common,
        }
    }

    pub fn activation_precision(&self) -> ConfigDataType {
        match self {
            EmbeddingConfig::Tied {
                precision,
                ..
            }
            | EmbeddingConfig::Untied {
                precision,
                ..
            } => *precision,
            EmbeddingConfig::MLXQuantizedTied {
                activation_precision,
                ..
            }
            | EmbeddingConfig::MLXQuantizedUntied {
                activation_precision,
                ..
            }
            | EmbeddingConfig::MLXSemiQuantizedUntied {
                activation_precision,
                ..
            } => *activation_precision,
        }
    }
}
