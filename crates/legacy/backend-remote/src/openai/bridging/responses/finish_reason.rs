use async_openai::types::responses::{Response, Status};
use shoji::types::session::chat::ChatReplyFinishReason;

pub fn build(response: &Response) -> Option<ChatReplyFinishReason> {
    match &response.status {
        Status::Completed | Status::Failed => Some(ChatReplyFinishReason::Stop),
        Status::Cancelled => Some(ChatReplyFinishReason::Cancelled),
        Status::Incomplete => match response.incomplete_details.as_ref().map(|details| details.reason.as_str()) {
            Some("max_output_tokens") => Some(ChatReplyFinishReason::Length),
            Some("content_filter") => Some(ChatReplyFinishReason::Rejected),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use async_openai::types::responses::{Response, ResponseStreamEvent};
    use shoji::types::session::chat::ChatReplyFinishReason;

    use super::build;
    use crate::openai::bridging::responses::stream_chunk;

    /// Builds a `Response` from the provider's wire shape rather than from the
    /// struct, so the field names the mapping depends on stay pinned to what the
    /// API actually sends.
    fn response(
        status: &str,
        incomplete_reason: Option<&str>,
    ) -> Response {
        let value = serde_json::json!({
            "id": "resp_test",
            "object": "response",
            "created_at": 1_700_000_000_u64,
            "model": "test-model",
            "output": [],
            "status": status,
            "incomplete_details": incomplete_reason.map(|reason| serde_json::json!({ "reason": reason })),
        });
        serde_json::from_value(value).expect("response fixture should deserialize")
    }

    #[test]
    fn terminal_statuses_map_to_stop() {
        assert_eq!(build(&response("completed", None)), Some(ChatReplyFinishReason::Stop));
        assert_eq!(build(&response("failed", None)), Some(ChatReplyFinishReason::Stop));
    }

    #[test]
    fn cancelled_maps_to_cancelled() {
        assert_eq!(build(&response("cancelled", None)), Some(ChatReplyFinishReason::Cancelled));
    }

    #[test]
    fn non_terminal_statuses_have_no_finish_reason() {
        assert_eq!(build(&response("in_progress", None)), None);
        assert_eq!(build(&response("queued", None)), None);
    }

    #[test]
    fn truncated_response_maps_to_length() {
        assert_eq!(build(&response("incomplete", Some("max_output_tokens"))), Some(ChatReplyFinishReason::Length));
    }

    #[test]
    fn content_filtered_response_maps_to_rejected() {
        assert_eq!(build(&response("incomplete", Some("content_filter"))), Some(ChatReplyFinishReason::Rejected));
    }

    #[test]
    fn unrecognised_incomplete_reason_has_no_finish_reason() {
        assert_eq!(build(&response("incomplete", None)), None);
        assert_eq!(build(&response("incomplete", Some("something_else"))), None);
    }

    /// Drives the real event path, so a truncated reply cannot silently surface as
    /// a natural stop again.
    #[test]
    fn incomplete_event_emits_length_finish_reason() {
        let event: ResponseStreamEvent = serde_json::from_value(serde_json::json!({
            "type": "response.incomplete",
            "sequence_number": 3,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1_700_000_000_u64,
                "model": "test-model",
                "output": [],
                "status": "incomplete",
                "incomplete_details": { "reason": "max_output_tokens" },
            },
        }))
        .expect("response.incomplete event fixture should deserialize");

        let chunk = stream_chunk::build(event).expect("a response.incomplete event yields a chunk");

        assert_eq!(chunk.finish_reason, Some(ChatReplyFinishReason::Length));
    }
}
