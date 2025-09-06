use serde::{Deserialize, Serialize};

use super::LanguageModelConfig;

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
    pub model_config: LanguageModelConfig,
}
