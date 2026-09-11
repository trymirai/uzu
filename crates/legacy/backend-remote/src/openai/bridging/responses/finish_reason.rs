use async_openai::types::responses::Status;
use shoji::types::session::chat::ChatReplyFinishReason;

pub fn build(status: &Status) -> Option<ChatReplyFinishReason> {
    match status {
        Status::Completed | Status::Failed => Some(ChatReplyFinishReason::Stop),
        Status::Cancelled => Some(ChatReplyFinishReason::Cancelled),
        _ => None,
    }
}
