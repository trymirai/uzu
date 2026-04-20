use async_openai::types::responses::Status;
use shoji::types::session::chat::FinishReason;

pub fn build(status: &Status) -> Option<FinishReason> {
    match status {
        Status::Completed | Status::Failed => Some(FinishReason::Stop),
        Status::Cancelled => Some(FinishReason::Cancelled),
        _ => None,
    }
}
