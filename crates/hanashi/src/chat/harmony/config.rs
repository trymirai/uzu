use serde::{Deserialize, Serialize};
use shoji::types::session::chat::ChatModelCapabilities;

use crate::chat::harmony::EncodingName;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    #[serde(rename = "name")]
    pub encoding_name: EncodingName,
}

impl Config {
    pub fn capabilities(&self) -> ChatModelCapabilities {
        match self.encoding_name {
            EncodingName::GptOss => ChatModelCapabilities {
                supports_reasoning: true,
                supports_disable_reasoning: false,
                supports_tools: true,
                supports_multiple_tool_calls: false,
                requires_tools: false,
            },
        }
    }
}
