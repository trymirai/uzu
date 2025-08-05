use serde::{Deserialize, Serialize};

use super::DecoderConfig;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ModelConfig {
    pub toolchain_version: String,
    pub vendor: String,
    pub name: String,
    pub model_config: DecoderConfig,
    pub tokenizer_file_names: Vec<String>,
}
