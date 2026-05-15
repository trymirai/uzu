use proc_macros::uzu_config;

use crate::{ConfigDataType, backends::common::gpu_types::QuantizationMode};

#[uzu_config]
pub struct QuantizationConfig {
    pub group_size: usize,
    pub weight_quantization_mode: QuantizationMode,
    pub activation_quantization_mode: Option<QuantizationMode>,
    pub activation_precision: ConfigDataType,
}

#[uzu_config]
#[serde(tag = "type")]
pub enum LinearConfig {
    #[serde(rename = "FullPrecisionLinearConfig")]
    FullPrecision {
        precision: ConfigDataType,
    },
    #[serde(rename = "GroupQuantizedLinearConfig")]
    Quantized(QuantizationConfig),
    #[serde(rename = "MLXQuantizedLinearConfig")]
    MLXQuantized(QuantizationConfig),
    #[serde(rename = "QLoRALinearConfig")]
    QLoRA {
        #[serde(flatten)]
        quantization: QuantizationConfig,
        lora_rank: usize,
        lora_scale: f32,
    },
    #[serde(rename = "RHTLinearWrapperConfig")]
    RHTLinearWrapper {
        block_size: usize,
        inner_config: Box<LinearConfig>,
    },
}

impl LinearConfig {
    pub fn activation_precision(&self) -> ConfigDataType {
        match self {
            LinearConfig::FullPrecision {
                precision,
            } => *precision,
            LinearConfig::Quantized(quantization) => quantization.activation_precision,
            LinearConfig::MLXQuantized(quantization) => quantization.activation_precision,
            LinearConfig::QLoRA {
                quantization,
                ..
            } => quantization.activation_precision,
            LinearConfig::RHTLinearWrapper {
                inner_config,
                ..
            } => inner_config.activation_precision(),
        }
    }
}
