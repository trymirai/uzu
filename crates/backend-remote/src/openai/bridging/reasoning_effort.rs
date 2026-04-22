use async_openai::types::chat::ReasoningEffort as OpenAIReasoningEffort;
use shoji::types::session::chat::ChatReasoningEffort;

pub fn build(effort: ChatReasoningEffort) -> OpenAIReasoningEffort {
    match effort {
        ChatReasoningEffort::Disabled => OpenAIReasoningEffort::None,
        ChatReasoningEffort::Default => OpenAIReasoningEffort::Medium,
        ChatReasoningEffort::Low => OpenAIReasoningEffort::Low,
        ChatReasoningEffort::Medium => OpenAIReasoningEffort::Medium,
        ChatReasoningEffort::High => OpenAIReasoningEffort::High,
    }
}
