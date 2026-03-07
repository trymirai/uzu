use std::{
    cell::Cell,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{SystemTime, UNIX_EPOCH},
};

use rocket::{
    Request, State,
    http::{ContentType, Header},
    post,
    response::{Responder, Response},
    serde::json::Json,
};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, ReadBuf};
use uuid::Uuid;
use uzu::session::{
    config::RunConfig,
    parameter::SamplingPolicy,
    types::{FinishReason, Input, Message, Output, Role, RunStats, Stats, StepStats, TotalStats},
};

use crate::server::{SessionState, SessionWrapper};

const LOG_PREVIEW_RESPONSE_BYTES: usize = 3000;
const LOG_PREVIEW_REQUEST_BYTES: usize = 10000;

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub content: String,
    pub role: Role,
}

#[derive(Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub max_completion_tokens: Option<u64>,
    pub stream: Option<bool>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatCompletionUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatCompletionTimings {
    pub prompt_n: u64,
    pub prompt_ms: f64,
    pub prompt_per_token_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_n: u64,
    pub predicted_ms: f64,
    pub predicted_per_token_ms: f64,
    pub predicted_per_second: f64,
}

fn make_usage_and_timings(stats: &Stats) -> (ChatCompletionUsage, ChatCompletionTimings) {
    let prompt_tokens = stats.total_stats.tokens_count_input;
    let completion_tokens = stats.total_stats.tokens_count_output;
    let prompt_ms = stats.prefill_stats.duration * 1000.0;
    let predicted_ms = stats.generate_stats.as_ref().map(|s| s.duration * 1000.0).unwrap_or(0.0);
    let usage = ChatCompletionUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    };
    let timings = ChatCompletionTimings {
        prompt_n: prompt_tokens,
        prompt_ms,
        prompt_per_token_ms: if prompt_tokens > 0 {
            prompt_ms / prompt_tokens as f64
        } else {
            0.0
        },
        prompt_per_second: stats.prefill_stats.tokens_per_second,
        predicted_n: completion_tokens,
        predicted_ms,
        predicted_per_token_ms: if completion_tokens > 0 {
            predicted_ms / completion_tokens as f64
        } else {
            0.0
        },
        predicted_per_second: stats.generate_stats.as_ref().map(|s| s.tokens_per_second).unwrap_or(0.0),
    };
    (usage, timings)
}

fn zero_usage_and_timings() -> (ChatCompletionUsage, ChatCompletionTimings) {
    (
        ChatCompletionUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
        ChatCompletionTimings {
            prompt_n: 0,
            prompt_ms: 0.0,
            prompt_per_token_ms: 0.0,
            prompt_per_second: 0.0,
            predicted_n: 0,
            predicted_ms: 0.0,
            predicted_per_token_ms: 0.0,
            predicted_per_second: 0.0,
        },
    )
}

fn zero_stats() -> Stats {
    Stats {
        prefill_stats: StepStats {
            duration: 0.0,
            suffix_length: 0,
            tokens_count: 0,
            tokens_per_second: 0.0,
            processed_tokens_per_second: 0.0,
            model_run: RunStats {
                count: 0,
                average_duration: 0.0,
            },
            run: None,
            speculator_proposed: 0,
            speculator_accepted: 0,
        },
        generate_stats: None,
        total_stats: TotalStats {
            duration: 0.0,
            tokens_count_input: 0,
            tokens_count_output: 0,
        },
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: ChatCompletionUsage,
    pub timings: ChatCompletionTimings,
    pub stats: Stats,
}

#[derive(Serialize)]
struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct StreamChoice {
    index: u32,
    delta: StreamDelta,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<StreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<ChatCompletionUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timings: Option<ChatCompletionTimings>,
}

struct StreamBody {
    rx: tokio::sync::mpsc::Receiver<String>,
    pending: Vec<u8>,
}

impl StreamBody {
    fn new(rx: tokio::sync::mpsc::Receiver<String>) -> Self {
        Self {
            rx,
            pending: Vec::new(),
        }
    }
}

impl AsyncRead for StreamBody {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        out: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let this = self.get_mut();

        // Drain any leftover bytes from the previous message first.
        if !this.pending.is_empty() {
            let n = this.pending.len().min(out.remaining());
            out.put_slice(&this.pending[..n]);
            this.pending.drain(..n);
            return Poll::Ready(Ok(()));
        }

        // Try to pull the next event string from the channel.
        match this.rx.poll_recv(cx) {
            Poll::Ready(Some(data)) => {
                let formatted = format!("data: {}\n\n", data);
                let bytes = formatted.into_bytes();
                let n = bytes.len().min(out.remaining());
                out.put_slice(&bytes[..n]);
                if n < bytes.len() {
                    this.pending = bytes[n..].to_vec();
                }
                Poll::Ready(Ok(()))
            },
            Poll::Ready(None) => Poll::Ready(Ok(())), // channel closed → EOF
            Poll::Pending => Poll::Pending,
        }
    }
}

struct AnyResponse(Response<'static>);

impl<'r> Responder<'r, 'static> for AnyResponse {
    fn respond_to(
        self,
        _: &'r Request<'_>,
    ) -> rocket::response::Result<'static> {
        Ok(self.0)
    }
}

enum ChatCompletionResult {
    Json(Json<ChatCompletionResponse>),
    Stream(AnyResponse),
}

impl<'r> Responder<'r, 'static> for ChatCompletionResult {
    fn respond_to(
        self,
        req: &'r Request<'_>,
    ) -> rocket::response::Result<'static> {
        match self {
            ChatCompletionResult::Json(j) => j.respond_to(req),
            ChatCompletionResult::Stream(a) => a.respond_to(req),
        }
    }
}

fn respond_stream(
    id: String,
    model_name: String,
    input: Input,
    run_config: RunConfig,
    created: i64,
    session_arc: Arc<SessionWrapper>,
    tx: tokio::sync::mpsc::Sender<String>,
) {
    let id_for_cb = id.clone();
    let model_for_cb = model_name.clone();
    let tx_for_cb = tx.clone();
    let prev_len = Cell::new(0usize);

    let result = {
        let mut session = session_arc.lock();
        session.run(
            input,
            run_config,
            Some(Box::new(move |output: Output| {
                let text = &output.text.original;
                let len = prev_len.get();
                let char_start = (len..=text.len()).find(|&i| text.is_char_boundary(i)).unwrap_or(text.len());
                if text.len() > char_start {
                    let delta = text[char_start..].to_string();
                    prev_len.set(text.len());
                    let chunk = ChatCompletionChunk {
                        id: id_for_cb.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_for_cb.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: StreamDelta {
                                content: Some(delta),
                            },
                            finish_reason: None,
                        }],
                        usage: None,
                        timings: None,
                    };
                    if let Ok(json) = serde_json::to_string(&chunk) {
                        if tx_for_cb.blocking_send(json).is_err() {
                            return false; // client disconnected, stop generation
                        }
                    }
                }
                true
            })),
        )
    };

    let (finish_reason, final_usage, final_timings) = match &result {
        Ok(output) => {
            let fr = output
                .finish_reason
                .as_ref()
                .map(|fr| format!("{:?}", fr).to_lowercase())
                .unwrap_or_else(|| "stop".to_string());
            let (usage, timings) = make_usage_and_timings(&output.stats);
            (fr, Some(usage), Some(timings))
        },
        Err(e) => {
            eprintln!("❌ [{}] Stream session run error: {}", id, e);
            let error_chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta {
                        content: Some(format!("Error: {}", e)),
                    },
                    finish_reason: None,
                }],
                usage: None,
                timings: None,
            };
            if let Ok(json) = serde_json::to_string(&error_chunk) {
                tx.blocking_send(json).ok();
            }
            let (usage, timings) = zero_usage_and_timings();
            ("error".to_string(), Some(usage), Some(timings))
        },
    };

    println!("📤 [{}] Stream finished, finish_reason={}", id, finish_reason);

    let final_chunk = ChatCompletionChunk {
        id,
        object: "chat.completion.chunk".to_string(),
        created,
        model: model_name,
        choices: vec![StreamChoice {
            index: 0,
            delta: StreamDelta {
                content: None,
            },
            finish_reason: Some(finish_reason),
        }],
        usage: final_usage,
        timings: final_timings,
    };
    if let Ok(json) = serde_json::to_string(&final_chunk) {
        tx.blocking_send(json).ok();
    }
    tx.blocking_send("[DONE]".to_string()).ok();
    // tx dropped here → channel closed → StreamBody reaches EOF
}

fn respond_blocking(
    id: String,
    model_name: String,
    input: Input,
    run_config: RunConfig,
    created: i64,
    session_wrapper: &SessionWrapper,
) -> ChatCompletionResult {
    let start_time = std::time::Instant::now();
    let mut session = session_wrapper.lock();

    let output = match session.run(input, run_config, None) {
        Ok(output) => output,
        Err(e) => {
            eprintln!("❌ [{}] Session run error: {}", id, e);
            let (usage, timings) = zero_usage_and_timings();
            let error_response = ChatCompletionResponse {
                id,
                object: "chat.completion".to_string(),
                created,
                model: model_name,
                choices: vec![ChatCompletionChoice {
                    index: 0,
                    message: ChatMessage {
                        content: format!("Error: {}", e),
                        role: Role::Assistant,
                    },
                    finish_reason: "error".to_string(),
                }],
                usage,
                timings,
                stats: zero_stats(),
            };
            return ChatCompletionResult::Json(Json(error_response));
        },
    };

    let processing_time = start_time.elapsed();

    println!("output: {:?}", &output);

    let Output {
        text,
        stats,
        finish_reason,
        ..
    } = output;
    let finish_reason_val = finish_reason.unwrap_or(FinishReason::Cancelled);

    println!("📤 [{}] Sending response:", id);
    println!("   Response length: {} chars", text.original.len());
    println!("   Finish reason: {:?}", finish_reason_val);
    println!("   Processing time: {:.3}s", processing_time.as_secs_f64());
    println!("   Tokens/second: {:.1}", stats.generate_stats.as_ref().map(|s| s.tokens_per_second).unwrap_or(0.0));

    let text = text.original;
    print!("   Response preview: ");
    if text.len() > LOG_PREVIEW_RESPONSE_BYTES {
        print!("{}", &text[..text.floor_char_boundary(LOG_PREVIEW_RESPONSE_BYTES)]);
        println!("...");
    } else {
        println!("{}", &text);
    }
    println!();

    let (usage, timings) = make_usage_and_timings(&stats);
    let response = ChatCompletionResponse {
        id,
        object: "chat.completion".to_string(),
        created,
        model: model_name,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                content: text,
                role: Role::Assistant,
            },
            finish_reason: format!("{:?}", finish_reason_val).to_lowercase(),
        }],
        usage,
        timings,
        stats,
    };

    ChatCompletionResult::Json(Json(response))
}

#[allow(private_interfaces)]
#[post("/chat/completions", format = "json", data = "<request>")]
pub async fn handle_chat_completions(
    request: Json<ChatCompletionRequest>,
    state: &State<SessionState>,
) -> ChatCompletionResult {
    let request = request.into_inner();
    let id = Uuid::new_v4().to_string();
    let is_stream = request.stream.unwrap_or(false);

    println!("📨 [{}] Incoming chat completion request (stream={}):", id, is_stream);
    println!("   Messages: {} message(s)", request.messages.len());
    println!("   Max tokens: {}", request.max_completion_tokens.unwrap_or(2048));

    for (i, msg) in request.messages.iter().enumerate() {
        let content_preview = if msg.content.len() > LOG_PREVIEW_REQUEST_BYTES {
            format!("{}...", &msg.content[..msg.content.floor_char_boundary(LOG_PREVIEW_REQUEST_BYTES)])
        } else {
            msg.content.clone()
        };
        println!("   Message {}: {:?} - {}", i + 1, msg.role, content_preview);
    }

    let model_name = state.model_name.clone();
    let tokens_limit = request.max_completion_tokens.unwrap_or(2048);
    let input = Input::Messages(
        request
            .messages
            .into_iter()
            .map(|m| Message {
                content: m.content,
                role: m.role,
                reasoning_content: None,
            })
            .collect(),
    );
    let run_config = RunConfig::new(tokens_limit, true, SamplingPolicy::Default, None);
    let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;

    let session_arc = Arc::clone(&state.session_wrapper);

    if is_stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(32);
        let tx_for_panic = tx.clone();
        tokio::task::spawn_blocking(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                respond_stream(id, model_name, input, run_config, created, session_arc, tx);
            }));
            if result.is_err() {
                // Ensure stream closes cleanly even after panic
                tx_for_panic.blocking_send("[DONE]".to_string()).ok();
            }
        });
        let response: Response<'static> = Response::build()
            .header(ContentType::new("text", "event-stream"))
            .header(Header::new("Cache-Control", "no-cache"))
            .header(Header::new("X-Accel-Buffering", "no"))
            .streamed_body(StreamBody::new(rx))
            .finalize();
        ChatCompletionResult::Stream(AnyResponse(response))
    } else {
        tokio::task::spawn_blocking(move || respond_blocking(id, model_name, input, run_config, created, &session_arc))
            .await
            .unwrap_or_else(|_| {
                let (usage, timings) = zero_usage_and_timings();
                ChatCompletionResult::Json(Json(ChatCompletionResponse {
                    id: "error".to_string(),
                    object: "chat.completion".to_string(),
                    created,
                    model: state.model_name.clone(),
                    choices: vec![],
                    usage,
                    timings,
                    stats: zero_stats(),
                }))
            })
    }
}
