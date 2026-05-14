use serde::{Deserialize, Serialize};

use super::ModelType;
use crate::backends::common::gpu_types::QuantizationMode;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ModelMetadata<T> {
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
