use rocket::{State, post, serde::json::Json};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;
use uzu::session::{
    session_input::SessionInput,
    session_message::{SessionMessage, SessionMessageRole},
    session_output::{
        SessionOutput, SessionOutputFinishReason, SessionOutputStats,
    },
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
    pub stats: SessionOutputStats,
}

#[post("/chat/completions", format = "json", data = "<request>")]
pub fn handle_chat_completions(
    request: Json<ChatCompletionRequest>,
    state: &State<SessionState>,
) -> Json<ChatCompletionResponse> {
    let request = request.into_inner();
    let id = Uuid::new_v4().to_string();

    println!("📨 [{}] Incoming chat completion request:", id);
    println!("   Messages: {} message(s)", request.messages.len());
    println!("   Max tokens: {}", request.max_tokens.unwrap_or(2048));
    println!("   System prompt key: {:?}", request.system_prompt_key);

    for (i, msg) in request.messages.iter().enumerate() {
        let content_preview = if msg.content.len() > 10000 {
            format!("{}...", &msg.content[..10000])
        } else {
            msg.content.clone()
        };
        println!("   Message {}: {} - {}", i + 1, msg.role, content_preview);
    }

    let model_name = state.model_name.clone();
    let system_prompt_key = request.system_prompt_key;
    let tokens_limit = request.max_tokens.unwrap_or(2048);
    let messages: Vec<SessionMessage> = request
        .messages
        .into_iter()
        .map(|m| SessionMessage {
            content: m.content,
            role: match m.role.as_str() {
                "system" => SessionMessageRole::System,
                "user" => SessionMessageRole::User,
                "assistant" => SessionMessageRole::Assistant,
                _ => SessionMessageRole::User,
            },
        })
        .collect();
    let input = SessionInput::Messages(messages);
    let run_config = SessionRunConfig::new(tokens_limit);

    let start_time = std::time::Instant::now();
    let mut session = state.session_wrapper.lock();

    let output = if let Some(key) = system_prompt_key {
        let context = state.cache.lock().unwrap().get(&key);
        let (output, new_context) = session.extend(input, context, run_config);
        state.cache.lock().unwrap().insert(key.clone(), new_context);
        output
    } else {
        session.run_with_context(input, None, run_config)
    };

    let processing_time = start_time.elapsed();

    println!("output: {:?}", &output);

    let SessionOutput {
        text,
        stats,
        finish_reason,
        ..
    } = output;

    let finish_reason_val =
        finish_reason.unwrap_or(SessionOutputFinishReason::Cancelled);

    println!("📤 [{}] Sending response:", id);
    println!("   Response length: {} chars", text.len());
    println!("   Finish reason: {:?}", finish_reason_val);
    println!("   Processing time: {:.3}s", processing_time.as_secs_f64());
    println!(
        "   Tokens/second: {:.1}",
        stats
            .generate_stats
            .as_ref()
            .map(|s| s.tokens_per_second)
            .unwrap_or(0.0)
    );

    print!("   Response preview: ");
    if text.len() > 3000 {
        print!("{}", &text[..3000]);
        println!("...");
    } else {
        println!("{}", &text);
    }
    println!();

    let response = ChatCompletionResponse {
        id: id.clone(),
        object: "chat.completion".to_string(),
        created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
            as i64,
        model: model_name,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                content: text,
                role: "assistant".to_string(),
            },
            finish_reason: format!("{:?}", finish_reason_val).to_lowercase(),
        }],
        stats,
    };

    Json(response)
}
