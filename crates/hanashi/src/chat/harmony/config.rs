use shoji::types::{model::HarmonyConfig, session::chat::ChatModelCapabilities};

pub fn harmony_config_capabilities(config: &HarmonyConfig) -> ChatModelCapabilities {
    match config {
        HarmonyConfig::GptOss => ChatModelCapabilities {
            supports_reasoning: true,
            supports_disable_reasoning: false,
            supports_tools: true,
            supports_multiple_tool_calls: false,
            requires_tools: false,
        },
    }
}
