use std::{
    pin::Pin,
    str::FromStr,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use rocket::{
    Request, State,
    futures::Stream,
    http::Status,
    post,
    response::{
        Responder, status,
        stream::{Event, EventStream},
    },
    serde::json::Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;
use uzu::{
    session::chat::{ChatSession, ChatSessionStreamChunk},
    types::{
        basic::{ReasoningEffort, SamplingMethod},
        session::chat::{ChatMessage, ChatReplyConfig, ChatReplyFinishReason, ChatReplyStats, ChatRole},
    },
};

use crate::server::ServerState;

#[derive(Serialize, Deserialize, Clone)]
pub struct OaiMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<OaiMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    // Raw value, not a typed string, so a bad reasoning_effort becomes our OpenAI 400
    // rather than Rocket's 422 at request extraction.
    pub reasoning_effort: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
}

#[derive(Serialize, Clone)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: OaiMessage,
    pub finish_reason: String,
}

#[derive(Serialize, Clone, Default)]
pub struct ChatCompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Serialize, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: ChatCompletionUsage,
}

#[derive(Serialize)]
struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
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
}

#[derive(Serialize)]
pub struct OaiErrorResponse {
    error: OaiError,
}

#[derive(Serialize)]
struct OaiError {
    message: String,
    #[serde(rename = "type")]
    kind: String,
    param: Option<String>,
    code: Option<String>,
}

pub enum ChatCompletionResult {
    Json(Json<ChatCompletionResponse>),
    Stream(EventStream<Pin<Box<dyn Stream<Item = Event> + Send>>>),
    Error(status::Custom<Json<OaiErrorResponse>>),
}

impl<'r> Responder<'r, 'r> for ChatCompletionResult {
    fn respond_to(
        self,
        request: &'r Request<'_>,
    ) -> rocket::response::Result<'r> {
        match self {
            ChatCompletionResult::Json(json) => json.respond_to(request),
            ChatCompletionResult::Stream(stream) => stream.respond_to(request),
            ChatCompletionResult::Error(error) => error.respond_to(request),
        }
    }
}

fn now_unix() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

fn to_chat_messages(
    messages: &[OaiMessage],
    reasoning_effort: Option<ReasoningEffort>,
) -> Vec<ChatMessage> {
    let mut messages = messages
        .iter()
        .map(|message| {
            let role = ChatRole::from_str(&message.role).unwrap_or(ChatRole::User {});
            ChatMessage::for_role(role).with_text(message.content.clone())
        })
        .collect::<Vec<_>>();
    if let (Some(effort), Some(message)) = (reasoning_effort, messages.last_mut()) {
        *message = message.with_reasoning_effort(effort);
    }
    messages
}

#[derive(Debug, PartialEq, Eq)]
enum RequestValidationError {
    UnsupportedReasoningEffort(&'static str),
    InvalidReasoningEffort(String),
}

impl RequestValidationError {
    fn message(&self) -> String {
        match self {
            RequestValidationError::UnsupportedReasoningEffort(value) => {
                format!("reasoning_effort `{value}` is recognized but not supported yet")
            },
            RequestValidationError::InvalidReasoningEffort(value) => {
                format!("reasoning_effort must be one of none, low, medium, high, disabled, default; got `{value}`")
            },
        }
    }

    fn code(&self) -> &'static str {
        match self {
            RequestValidationError::UnsupportedReasoningEffort(_) => "unsupported_reasoning_effort",
            RequestValidationError::InvalidReasoningEffort(_) => "invalid_reasoning_effort",
        }
    }

    fn param(&self) -> &'static str {
        "reasoning_effort"
    }
}

fn request_error_response(error: RequestValidationError) -> ChatCompletionResult {
    ChatCompletionResult::Error(status::Custom(
        Status::BadRequest,
        Json(OaiErrorResponse {
            error: OaiError {
                message: error.message(),
                kind: "invalid_request_error".to_string(),
                param: Some(error.param().to_string()),
                code: Some(error.code().to_string()),
            },
        }),
    ))
}

fn parse_reasoning_effort(
    value: Option<&serde_json::Value>
) -> Result<Option<ReasoningEffort>, RequestValidationError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Some(value) = value.as_str() else {
        return Err(RequestValidationError::InvalidReasoningEffort(value.to_string()));
    };
    match value {
        "none" | "disabled" => Ok(Some(ReasoningEffort::Disabled)),
        "default" => Ok(Some(ReasoningEffort::Default)),
        // Local chat templates currently expose a thinking on/off control. Keep the requested
        // level in the message so backends that can honor levels may do so, while local backends
        // treat all non-disabled levels as thinking enabled.
        "low" => Ok(Some(ReasoningEffort::Low)),
        "medium" => Ok(Some(ReasoningEffort::Medium)),
        "high" => Ok(Some(ReasoningEffort::High)),
        "minimal" => Err(RequestValidationError::UnsupportedReasoningEffort("minimal")),
        "xhigh" => Err(RequestValidationError::UnsupportedReasoningEffort("xhigh")),
        other => Err(RequestValidationError::InvalidReasoningEffort(other.to_string())),
    }
}

fn build_reply_config(request: &ChatCompletionRequest) -> ChatReplyConfig {
    let token_limit = request.max_completion_tokens.or(request.max_tokens);
    let config = ChatReplyConfig::default().with_token_limit(token_limit);

    if request.temperature.is_some_and(|temperature| temperature <= 0.0) {
        return config.with_sampling_method(SamplingMethod::Greedy {});
    } else if request.temperature.is_some() || request.top_p.is_some() || request.top_k.is_some() {
        return config.with_sampling_method(SamplingMethod::Stochastic {
            temperature: request.temperature,
            top_k: request.top_k,
            top_p: request.top_p,
            min_p: None,
            repetition_penalty: None,
            suffix_repetition_length: None,
        });
    }

    config
}

fn map_finish_reason(finish_reason: &ChatReplyFinishReason) -> String {
    match finish_reason {
        ChatReplyFinishReason::Stop | ChatReplyFinishReason::Cancelled => "stop",
        ChatReplyFinishReason::Length | ChatReplyFinishReason::ContextLimitReached => "length",
        ChatReplyFinishReason::ToolCalls => "tool_calls",
        ChatReplyFinishReason::Rejected => "content_filter",
    }
    .to_string()
}

fn usage_from_stats(stats: &ChatReplyStats) -> ChatCompletionUsage {
    let prompt_tokens = stats.tokens_count_input.unwrap_or(0);
    let completion_tokens = stats.tokens_count_output.unwrap_or(0);
    ChatCompletionUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    }
}

fn error_response(
    id: String,
    model: String,
    created: i64,
    message: &str,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id,
        object: "chat.completion".to_string(),
        created,
        model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: OaiMessage {
                role: "assistant".to_string(),
                content: format!("Error: {message}"),
            },
            finish_reason: "stop".to_string(),
        }],
        usage: ChatCompletionUsage::default(),
    }
}

fn chunk_json(
    id: &str,
    model: &str,
    created: i64,
    delta: StreamDelta,
    finish_reason: Option<String>,
    usage: Option<ChatCompletionUsage>,
) -> String {
    let chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![StreamChoice {
            index: 0,
            delta,
            finish_reason,
        }],
        usage,
    };
    serde_json::to_string(&chunk).unwrap_or_default()
}

async fn run_blocking(
    session: Arc<Mutex<ChatSession>>,
    messages: Vec<ChatMessage>,
    config: ChatReplyConfig,
    id: String,
    model: String,
    created: i64,
) -> ChatCompletionResponse {
    let session = session.lock().await;
    if let Err(error) = session.reset().await {
        return error_response(id, model, created, &error.to_string());
    }

    match session.reply(messages, config).await {
        Ok(replies) => match replies.last() {
            Some(reply) => ChatCompletionResponse {
                id,
                object: "chat.completion".to_string(),
                created,
                model,
                choices: vec![ChatCompletionChoice {
                    index: 0,
                    message: OaiMessage {
                        role: "assistant".to_string(),
                        content: reply.message.text().unwrap_or_default(),
                    },
                    finish_reason: reply
                        .finish_reason
                        .as_ref()
                        .map(map_finish_reason)
                        .unwrap_or_else(|| "stop".to_string()),
                }],
                usage: usage_from_stats(&reply.stats),
            },
            None => error_response(id, model, created, "No response generated"),
        },
        Err(error) => error_response(id, model, created, &error.to_string()),
    }
}

async fn run_stream(
    session: Arc<Mutex<ChatSession>>,
    messages: Vec<ChatMessage>,
    config: ChatReplyConfig,
    id: String,
    model: String,
    created: i64,
    sender: mpsc::UnboundedSender<Event>,
) {
    let session = session.lock().await;
    if let Err(error) = session.reset().await {
        let _ = sender.send(Event::data(chunk_json(
            &id,
            &model,
            created,
            StreamDelta {
                role: None,
                content: Some(format!("Error: {error}")),
            },
            Some("stop".to_string()),
            None,
        )));
        let _ = sender.send(Event::data("[DONE]"));
        return;
    }

    let _ = sender.send(Event::data(chunk_json(
        &id,
        &model,
        created,
        StreamDelta {
            role: Some("assistant".to_string()),
            content: None,
        },
        None,
        None,
    )));

    let stream = session.reply_with_stream(messages, config).await;
    let mut emitted = 0usize;
    let mut finish_reason = "stop".to_string();
    let mut usage = ChatCompletionUsage::default();
    let mut errored = false;

    while let Some(chunk) = stream.next().await {
        match chunk {
            ChatSessionStreamChunk::Replies {
                replies,
            } => {
                let Some(reply) = replies.last() else {
                    continue;
                };
                let text = reply.message.text().unwrap_or_default();
                let start = (emitted..=text.len()).find(|&index| text.is_char_boundary(index)).unwrap_or(text.len());
                if text.len() > start {
                    let delta = text[start..].to_string();
                    emitted = text.len();
                    let sent = sender.send(Event::data(chunk_json(
                        &id,
                        &model,
                        created,
                        StreamDelta {
                            role: None,
                            content: Some(delta),
                        },
                        None,
                        None,
                    )));
                    if sent.is_err() {
                        return;
                    }
                }
                if let Some(reason) = &reply.finish_reason {
                    finish_reason = map_finish_reason(reason);
                }
                usage = usage_from_stats(&reply.stats);
            },
            ChatSessionStreamChunk::Error {
                error,
            } => {
                errored = true;
                let _ = sender.send(Event::data(chunk_json(
                    &id,
                    &model,
                    created,
                    StreamDelta {
                        role: None,
                        content: Some(format!("Error: {error}")),
                    },
                    Some("stop".to_string()),
                    None,
                )));
                break;
            },
        }
    }

    if !errored {
        let _ = sender.send(Event::data(chunk_json(
            &id,
            &model,
            created,
            StreamDelta {
                role: None,
                content: None,
            },
            Some(finish_reason),
            Some(usage),
        )));
    }
    let _ = sender.send(Event::data("[DONE]"));
}

#[allow(private_interfaces)]
#[post("/chat/completions", format = "json", data = "<request>")]
pub async fn handle_chat_completions(
    request: Json<ChatCompletionRequest>,
    state: &State<ServerState>,
) -> ChatCompletionResult {
    let request = request.into_inner();
    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let created = now_unix();
    let model = state.model_name.clone();
    let is_stream = request.stream.unwrap_or(false);

    let config = build_reply_config(&request);
    let reasoning_effort = match parse_reasoning_effort(request.reasoning_effort.as_ref()) {
        Ok(reasoning_effort) => reasoning_effort,
        Err(error) => return request_error_response(error),
    };
    let messages = to_chat_messages(&request.messages, reasoning_effort);

    if is_stream {
        let session = Arc::clone(&state.session);
        let (sender, receiver) = mpsc::unbounded_channel::<Event>();
        rocket::tokio::spawn(run_stream(session, messages, config, id, model, created, sender));
        let body: Pin<Box<dyn Stream<Item = Event> + Send>> = Box::pin(UnboundedReceiverStream::new(receiver));
        ChatCompletionResult::Stream(EventStream::from(body))
    } else {
        let session = Arc::clone(&state.session);
        let response = run_blocking(session, messages, config, id, model, created).await;
        ChatCompletionResult::Json(Json(response))
    }
}

#[cfg(test)]
mod tests {
    use uzu::types::session::chat::ChatMessageList;

    use super::*;

    fn request(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).expect("valid request json")
    }

    fn reply_config(json: &str) -> ChatReplyConfig {
        build_reply_config(&request(json))
    }

    fn chat_messages(json: &str) -> Vec<ChatMessage> {
        let request = request(json);
        to_chat_messages(
            &request.messages,
            parse_reasoning_effort(request.reasoning_effort.as_ref()).expect("valid reasoning_effort"),
        )
    }

    // Test-only route returning a reasoning_effort validation error, used to exercise the actual
    // Rocket response layer. Defined at module level so the `rocket::get` macro stays local.
    #[rocket::get("/err")]
    fn err_route() -> ChatCompletionResult {
        request_error_response(RequestValidationError::InvalidReasoningEffort("bad".to_string()))
    }

    #[test]
    fn error_responder_yields_http_400_with_openai_body() {
        let client = rocket::local::blocking::Client::tracked(rocket::build().mount("/", rocket::routes![err_route]))
            .expect("rocket client");
        let response = client.get("/err").dispatch();

        assert_eq!(response.status(), Status::BadRequest);
        let body: serde_json::Value = response.into_json().expect("json error body");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "reasoning_effort");
        assert_eq!(body["error"]["code"], "invalid_reasoning_effort");
        assert!(
            body["error"]["message"].as_str().is_some_and(|message| !message.is_empty()),
            "expected a non-empty error message, got {body}"
        );
    }

    #[test]
    fn reasoning_effort_is_optional() {
        let messages = chat_messages(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(messages.reasoning_effort(), None);
    }

    #[test]
    fn reasoning_effort_applies_to_latest_message() {
        let messages = chat_messages(
            r#"{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"}],"reasoning_effort":"none"}"#,
        );

        assert_eq!(messages.reasoning_effort(), Some(ReasoningEffort::Disabled));
        assert_eq!(messages.first().and_then(ChatMessage::reasoning_effort), None);
        assert_eq!(messages.last().and_then(ChatMessage::reasoning_effort), Some(ReasoningEffort::Disabled));
    }

    #[test]
    fn reasoning_effort_accepts_openai_values_and_uzu_aliases() {
        for (value, expected) in [
            ("none", ReasoningEffort::Disabled),
            ("disabled", ReasoningEffort::Disabled),
            ("default", ReasoningEffort::Default),
            ("low", ReasoningEffort::Low),
            ("medium", ReasoningEffort::Medium),
            ("high", ReasoningEffort::High),
        ] {
            let request = request(&format!(r#"{{"messages":[],"reasoning_effort":"{value}"}}"#));
            assert_eq!(parse_reasoning_effort(request.reasoning_effort.as_ref()), Ok(Some(expected)));
        }
    }

    #[test]
    fn recognized_unsupported_reasoning_effort_is_request_error() {
        for value in ["minimal", "xhigh"] {
            let request = request(&format!(r#"{{"messages":[],"reasoning_effort":"{value}"}}"#));
            let error = parse_reasoning_effort(request.reasoning_effort.as_ref())
                .expect_err("unsupported reasoning_effort should be rejected");
            assert_eq!(error, RequestValidationError::UnsupportedReasoningEffort(value));
            assert_eq!(error.param(), "reasoning_effort");
            assert_eq!(error.code(), "unsupported_reasoning_effort");
        }
    }

    #[test]
    fn invalid_reasoning_effort_is_request_error() {
        let request = request(r#"{"messages":[],"reasoning_effort":"maximum"}"#);
        let error = parse_reasoning_effort(request.reasoning_effort.as_ref())
            .expect_err("invalid reasoning_effort should be rejected");
        assert_eq!(error, RequestValidationError::InvalidReasoningEffort("maximum".to_string()));
        assert_eq!(error.param(), "reasoning_effort");
        assert_eq!(error.code(), "invalid_reasoning_effort");
    }

    #[test]
    fn malformed_reasoning_effort_passes_json_extraction() {
        for body in
            [r#"{"messages":[],"reasoning_effort":123}"#, r#"{"messages":[],"reasoning_effort":{"level":"disabled"}}"#]
        {
            let request = serde_json::from_str::<ChatCompletionRequest>(body)
                .unwrap_or_else(|error| panic!("expected {body} to pass extraction, got {error}"));
            let error = parse_reasoning_effort(request.reasoning_effort.as_ref())
                .expect_err("malformed reasoning_effort should be rejected");
            assert_eq!(error.param(), "reasoning_effort");
            assert_eq!(error.code(), "invalid_reasoning_effort");
        }
    }

    #[test]
    fn reasoning_effort_composes_with_sampling_options() {
        let stochastic =
            reply_config(r#"{"messages":[],"temperature":0.7,"top_p":0.9,"top_k":40,"reasoning_effort":"none"}"#);
        let messages = chat_messages(
            r#"{"messages":[{"role":"user","content":"json please"}],"temperature":0.7,"top_p":0.9,"top_k":40,"reasoning_effort":"none"}"#,
        );
        assert_eq!(messages.reasoning_effort(), Some(ReasoningEffort::Disabled));
        assert_eq!(
            stochastic.sampling_policy,
            uzu::types::basic::SamplingPolicy::Custom {
                method: SamplingMethod::Stochastic {
                    temperature: Some(0.7),
                    top_k: Some(40),
                    top_p: Some(0.9),
                    min_p: None,
                    repetition_penalty: None,
                    suffix_repetition_length: None,
                },
            }
        );

        let greedy = reply_config(r#"{"messages":[],"temperature":0,"reasoning_effort":"none"}"#);
        assert_eq!(
            greedy.sampling_policy,
            uzu::types::basic::SamplingPolicy::Custom {
                method: SamplingMethod::Greedy {},
            }
        );
    }
}
