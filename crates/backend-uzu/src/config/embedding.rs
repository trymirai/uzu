use proc_macros::uzu_config;

use crate::{DataType, backends::common::gpu_types::QuantizationMode};

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
        precision: DataType,
    },
    #[serde(rename = "UntiedEmbeddingConfig")]
    Untied {
        #[serde(flatten)]
        common: EmbeddingConfigCommon,
        precision: DataType,
    },
    #[serde(rename = "ScaleBiasQuantizedTiedEmbeddingConfig")]
    ScaleBiasQuantizedTied {
        #[serde(flatten)]
        common: EmbeddingConfigCommon,
        group_size: usize,
        embedding_quantization_mode: QuantizationMode,
        activation_quantization_mode: Option<QuantizationMode>,
        activation_precision: DataType,
    },
    #[serde(rename = "ScaleBiasQuantizedUntiedEmbeddingConfig")]
    ScaleBiasQuantizedUntied {
        #[serde(flatten)]
        common: EmbeddingConfigCommon,
        group_size: usize,
        embedding_quantization_mode: QuantizationMode,
        activation_quantization_mode: Option<QuantizationMode>,
        activation_precision: DataType,
    },
    #[serde(rename = "ScaleBiasSemiQuantizedUntiedEmbeddingConfig")]
    ScaleBiasSemiQuantizedUntied {
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
            | EmbeddingConfig::ScaleBiasQuantizedTied {
                common,
                ..
            }
            | EmbeddingConfig::ScaleBiasQuantizedUntied {
                common,
                ..
            }
            | EmbeddingConfig::ScaleBiasSemiQuantizedUntied {
                common,
                ..
            } => common,
        }
    }

    pub fn activation_precision(&self) -> DataType {
        match self {
            EmbeddingConfig::Tied {
                precision,
                ..
            }
            | EmbeddingConfig::Untied {
                precision,
                ..
            } => *precision,
            EmbeddingConfig::ScaleBiasQuantizedTied {
                activation_precision,
                ..
            }
            | EmbeddingConfig::ScaleBiasQuantizedUntied {
                activation_precision,
                ..
            }
            | EmbeddingConfig::ScaleBiasSemiQuantizedUntied {
                activation_precision,
                ..
            } => *activation_precision,
        }
    }
}
