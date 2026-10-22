use serde::{Deserialize, Serialize};
use shoji::types::{basic::ReasoningEffort, session::chat::ChatModelCapabilities};

use crate::chat::harmony::bridging::FromHarmony;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum HarmonyConfig {
    #[serde(rename = "gpt-oss")]
    GptOss,
}

impl HarmonyConfig {
    pub fn default_reasoning_effort(&self) -> Option<ReasoningEffort> {
        match self {
            HarmonyConfig::GptOss => {
                openai_harmony::chat::SystemContent::default().reasoning_effort.map(ReasoningEffort::from_harmony)
            },
        }
    }

    pub fn capabilities(&self) -> ChatModelCapabilities {
        match self {
            HarmonyConfig::GptOss => ChatModelCapabilities {
                supports_reasoning: true,
                supports_disable_reasoning: false,
                reasoning_efforts: vec![ReasoningEffort::Low, ReasoningEffort::Medium, ReasoningEffort::High],
                supports_tools: true,
                supports_multiple_tool_calls: false,
                requires_tools: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt_oss_exposes_supported_efforts_and_native_default() {
        let config = HarmonyConfig::GptOss;
        assert_eq!(
            config.capabilities().reasoning_efforts,
            vec![ReasoningEffort::Low, ReasoningEffort::Medium, ReasoningEffort::High]
        );
        assert_eq!(config.default_reasoning_effort(), Some(ReasoningEffort::Medium));
    }
}
