use async_openai::types::chat::FinishReason as OpenAiFinishReason;
use shoji::types::session::chat::ChatFinishReason;

pub fn build(reason: OpenAiFinishReason) -> ChatFinishReason {
    match reason {
        OpenAiFinishReason::Stop => ChatFinishReason::Stop,
        OpenAiFinishReason::Length => ChatFinishReason::Length,
        OpenAiFinishReason::ToolCalls => ChatFinishReason::ToolCalls,
        OpenAiFinishReason::ContentFilter => ChatFinishReason::Rejected,
        OpenAiFinishReason::FunctionCall => ChatFinishReason::ToolCalls,
    }
}
