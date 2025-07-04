use rocket::{post, serde::json::Json};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use uzu::session::{
    session_config::SessionRunConfig,
    session_input::SessionInput,
    session_message::{SessionMessage, SessionMessageRole},
    session_output::{
        SessionOutput, SessionOutputFinishReason, SessionOutputStats,
    },
};

use crate::server::SessionState;

#[derive(Deserialize)]
pub struct ChatCompletionsRequest {
    messages: Vec<SessionMessage>,
    max_completion_tokens: Option<usize>,
}

#[derive(Serialize)]
pub struct ChatCompletionsResponseChoice {
    index: usize,
    message: SessionMessage,
    finish_reason: SessionOutputFinishReason,
}

#[derive(Serialize)]
pub struct ChatCompletionsResponse {
    id: String,
    model: String,
    choices: Vec<ChatCompletionsResponseChoice>,
    stats: SessionOutputStats,
}

#[post("/chat/completions", data = "<request>")]
pub fn handle_chat_completions(
    request: Json<ChatCompletionsRequest>,
    state: &rocket::State<SessionState>,
) -> Json<ChatCompletionsResponse> {
    let id = Uuid::new_v4().to_string();

    let tokens_limit = request.max_completion_tokens.unwrap_or(128) as u64;
    let input = SessionInput::Messages(request.messages.clone());
    let run_config = SessionRunConfig::new(tokens_limit);

    let mut session = state.session_wrapper.lock();
    let output = session.run(
        input,
        run_config,
        Some(|_: SessionOutput| {
            return true;
        }),
    );

    let choice = ChatCompletionsResponseChoice {
        index: 0,
        message: SessionMessage {
            role: SessionMessageRole::Assistant,
            content: output.text,
        },
        finish_reason: output
            .finish_reason
            .unwrap_or(SessionOutputFinishReason::Cancelled),
    };

    let response = ChatCompletionsResponse {
        id: id,
        model: state.model_name.clone(),
        choices: vec![choice],
        stats: output.stats,
    };

    Json(response)
}
