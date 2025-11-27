use serde::{Deserialize, Serialize};

use crate::{ModelConfig, ModelType};

fn default_model_type() -> ModelType {
    ModelType::LanguageModel
}

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
    #[serde(default = "default_model_type")]
    pub model_type: ModelType,
    pub model_config: ModelConfig,
}
