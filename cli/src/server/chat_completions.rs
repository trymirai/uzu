use std::time::{SystemTime, UNIX_EPOCH};
use rocket::{post, serde::json::Json, State};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use uzu::session::{
    session_input::SessionInput,
    session_message::SessionMessage,
    session_output::SessionOutputFinishReason,
    session_run_config::SessionRunConfig,
};

use crate::server::SessionState;

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub content: String,
    pub role: String,
}

#[derive(Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub model: String,
    pub max_tokens: Option<u64>,
    pub system_prompt_key: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
}

#[post("/chat/completions", format = "json", data = "<request>")]
pub fn handle_chat_completions(
    request: Json<ChatCompletionRequest>,
    state: &State<SessionState>,
) -> Json<ChatCompletionResponse> {
    let request = request.into_inner();
    let model_name = state.model_name.clone();
    let system_prompt_key = request.system_prompt_key;
    let tokens_limit = request.max_tokens.unwrap_or(2048);
    let messages: Vec<SessionMessage> = request
        .messages
        .into_iter()
        .map(|m| SessionMessage {
            content: m.content,
            role: match m.role.as_str() {
                "system" => uzu::session::session_message::SessionMessageRole::System,
                "user" => uzu::session::session_message::SessionMessageRole::User,
                "assistant" => uzu::session::session_message::SessionMessageRole::Assistant,
                _ => uzu::session::session_message::SessionMessageRole::User,
            },
        })
        .collect();
    let input = SessionInput::Messages(messages);
    let mut session = state.session_wrapper.lock();

    let output = if let Some(key) = system_prompt_key {
        let context = state.cache.lock().unwrap().get(&key);
        let (output, new_context) = session.extend(
            input,
            context,
            SessionRunConfig::new(tokens_limit),
        );
        state.cache.lock().unwrap().insert(key.clone(), new_context);
        output
    } else {
        session.run_with_context(
            input,
            None,
            SessionRunConfig::new(tokens_limit),
        )
    };

    let response = ChatCompletionResponse {
        id: Uuid::new_v4().to_string(),
        object: "chat.completion".to_string(),
        created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64,
        model: model_name,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                content: output.text,
                role: "assistant".to_string(),
            },
            finish_reason: output
                .finish_reason
                .map(|fr| format!("{:?}", fr).to_lowercase())
                .unwrap_or_else(|| "stop".to_string()),
        }],
    };

    Json(response)
}
