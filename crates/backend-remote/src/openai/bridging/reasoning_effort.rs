use async_openai::types::chat::ReasoningEffort as OpenAIReasoningEffort;
use shoji::types::session::chat::ReasoningEffort;

pub fn build(effort: ReasoningEffort) -> OpenAIReasoningEffort {
    match effort {
        ReasoningEffort::Disabled => OpenAIReasoningEffort::None,
        ReasoningEffort::Default => OpenAIReasoningEffort::Medium,
        ReasoningEffort::Low => OpenAIReasoningEffort::Low,
        ReasoningEffort::Medium => OpenAIReasoningEffort::Medium,
        ReasoningEffort::High => OpenAIReasoningEffort::High,
    }
}
