use serde::{Deserialize, Serialize};

use crate::{ModelConfig, ModelType};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ModelMetadata {
    pub toolchain_version: String,
    pub vendor: String,
    pub family: String,
    pub name: String,
    pub size: String,
    pub quantization: Option<String>,
    pub repo: String,
    pub use_cases: Vec<String>,
    pub model_type: ModelType,
    pub model_config: ModelConfig,
}
