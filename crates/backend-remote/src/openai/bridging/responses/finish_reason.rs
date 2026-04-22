use async_openai::types::responses::Status;
use shoji::types::session::chat::ChatFinishReason;

pub fn build(status: &Status) -> Option<ChatFinishReason> {
    match status {
        Status::Completed | Status::Failed => Some(ChatFinishReason::Stop),
        Status::Cancelled => Some(ChatFinishReason::Cancelled),
        _ => None,
    }
}
