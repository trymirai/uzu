use proc_macros::uzu_config;

use super::{ClassifierModelConfig, LanguageModelConfig, ModelType};
#[cfg(metal_backend)]
use super::TtsModelConfig;
use crate::{backends::common::gpu_types::QuantizationMode, utils::strict_serde::DeserializeStrictOwned};

#[uzu_config]
#[serde(untagged)]
pub enum ModelConfig {
    LanguageModel(LanguageModelConfig),
    ClassifierModel(ClassifierModelConfig),
    #[cfg(metal_backend)]
    TtsModel(TtsModelConfig),
}

#[uzu_config]
pub struct ModelMetadata<T: DeserializeStrictOwned> {
    pub toolchain_version: String,
    pub vendor: String,
    pub family: String,
    pub name: String,
    pub size: String,
    pub quantization: Option<QuantizationMode>,
    pub repo: String,
    pub use_cases: Vec<String>,
    pub model_type: ModelType,
    pub model_config: T,
    pub grammar_start_tokens: Vec<String>,
}
