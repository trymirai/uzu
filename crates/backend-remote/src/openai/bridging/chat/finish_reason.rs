use async_openai::types::chat::FinishReason as ChatFinishReason;
use shoji::types::session::chat::FinishReason;

pub fn build(reason: ChatFinishReason) -> FinishReason {
    match reason {
        ChatFinishReason::Stop => FinishReason::Stop,
        ChatFinishReason::Length => FinishReason::Length,
        ChatFinishReason::ToolCalls => FinishReason::ToolCalls,
        ChatFinishReason::ContentFilter => FinishReason::Rejected,
        ChatFinishReason::FunctionCall => FinishReason::ToolCalls,
    }
}
