use rocket::{post, serde::json::Json};
use serde::{Deserialize, Serialize};
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

    println!("📨 [{}] Incoming chat completion request:", id);
    println!("   Messages: {} message(s)", request.messages.len());
    println!("   Max tokens: {}", request.max_completion_tokens.unwrap_or(512));
    for (i, msg) in request.messages.iter().enumerate() {
        let content_preview = if msg.content.len() > 10000 {
            format!("{}...", &msg.content[..10000])
        } else {
            msg.content.clone()
        };
        println!("   Message {}: {:?} - {}", i + 1, msg.role, content_preview);
    }

    let tokens_limit = request.max_completion_tokens.unwrap_or(512) as u64;
    let input = SessionInput::Messages(request.messages.clone());
    let run_config = SessionRunConfig::new(tokens_limit);

    let start_time = std::time::Instant::now();
    let mut session = state.session_wrapper.lock();
    let output = session.run(
        input,
        run_config,
        Some(|_: SessionOutput| {
            return true;
        }),
    );
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

    let choice = ChatCompletionsResponseChoice {
        index: 0,
        message: SessionMessage {
            role: SessionMessageRole::Assistant,
            content: text,
        },
        finish_reason: finish_reason_val,
    };

    let response = ChatCompletionsResponse {
        id: id.clone(),
        model: state.model_name.clone(),
        choices: vec![choice],
        stats,
    };

    Json(response)
}
