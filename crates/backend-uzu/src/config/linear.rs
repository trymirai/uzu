use proc_macros::uzu_config;

use crate::{DataType, backends::common::gpu_types::QuantizationMode};

#[uzu_config]
pub struct QuantizationConfig {
    pub group_size: usize,
    pub weight_quantization_mode: QuantizationMode,
    pub activation_quantization_mode: Option<QuantizationMode>,
    pub activation_precision: DataType,
}

#[uzu_config]
#[serde(tag = "type")]
pub enum LinearConfig {
    #[serde(rename = "FullPrecisionLinearConfig")]
    FullPrecision {
        precision: DataType,
    },
    #[serde(rename = "GroupQuantizedLinearConfig")]
    Quantized(QuantizationConfig),
    #[serde(rename = "MLXQuantizedLinearConfig")]
    ScaleBiasQuantized(QuantizationConfig),
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
    pub fn activation_precision(&self) -> DataType {
        match self {
            LinearConfig::FullPrecision {
                precision,
            } => *precision,
            LinearConfig::Quantized(quantization) => quantization.activation_precision,
            LinearConfig::ScaleBiasQuantized(quantization) => quantization.activation_precision,
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
