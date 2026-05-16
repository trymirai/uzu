use proc_macros::uzu_config;

use crate::{ConfigDataType, backends::common::gpu_types::QuantizationMode};

#[uzu_config]
pub struct EmbeddingConfigCommon {
    pub input_scale: Option<f32>,
    pub logit_soft_cap: Option<f32>,
}

#[uzu_config]
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
