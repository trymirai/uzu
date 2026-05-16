use serde::{Deserialize, Serialize};

use super::ModelType;
use crate::backends::common::gpu_types::QuantizationMode;

// TODO: Switch to #[uzu_config] once tts configs are migrated too
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelMetadata<T> {
    pub toolchain_version: String,
    pub vendor: String,
    pub family: String,
    pub name: String,
    pub size: String,
    #[serde(deserialize_with = "crate::utils::strict_serde::required")]
    pub quantization: Option<QuantizationMode>,
    pub repo: String,
    pub use_cases: Vec<String>,
    pub model_type: ModelType,
    pub model_config: T,
    pub grammar_start_tokens: Vec<String>,
}
