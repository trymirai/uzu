use async_openai::types::chat::FinishReason as OpenAiFinishReason;
use shoji::types::session::chat::ChatReplyFinishReason;

pub fn build(reason: OpenAiFinishReason) -> ChatReplyFinishReason {
    match reason {
        OpenAiFinishReason::Stop => ChatReplyFinishReason::Stop,
        OpenAiFinishReason::Length => ChatReplyFinishReason::Length,
        OpenAiFinishReason::ToolCalls => ChatReplyFinishReason::ToolCalls,
        OpenAiFinishReason::ContentFilter => ChatReplyFinishReason::Rejected,
        OpenAiFinishReason::FunctionCall => ChatReplyFinishReason::ToolCalls,
    }
}
