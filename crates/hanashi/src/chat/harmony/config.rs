use serde::{Deserialize, Serialize};
use shoji::types::session::chat::ChatModelCapabilities;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum HarmonyConfig {
    #[serde(rename = "gpt-oss")]
    GptOss,
}

impl HarmonyConfig {
    pub fn capabilities(&self) -> ChatModelCapabilities {
        match self {
            HarmonyConfig::GptOss => ChatModelCapabilities {
                supports_reasoning: true,
                supports_disable_reasoning: false,
                supports_tools: true,
                supports_multiple_tool_calls: false,
                requires_tools: false,
            },
        }
    }
}
