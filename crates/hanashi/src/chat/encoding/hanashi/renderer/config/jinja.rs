use serde::{Deserialize, Serialize};

use crate::chat::encoding::hanashi::renderer::config::JinjaFunction;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JinjaConfig {
    pub template: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub required_functions: Vec<JinjaFunction>,
    pub preamble_control_key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bos_token_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eos_token_key: Option<String>,
}
