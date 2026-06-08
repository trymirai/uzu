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
        basic::{Grammar, SamplingMethod},
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
    // Captured as a raw value (not a typed `ResponseFormat`) so the handler can reject it with an
    // OpenAI-style 400 — both unrecognized/malformed objects and the recognized-but-unsupported
    // `json_schema` form — instead of failing Rocket's request extraction with a 422.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
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

fn to_chat_messages(messages: &[OaiMessage]) -> Vec<ChatMessage> {
    messages
        .iter()
        .map(|message| {
            let role = ChatRole::from_str(&message.role).unwrap_or(ChatRole::User {});
            ChatMessage::for_role(role).with_text(message.content.clone())
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
enum ResponseFormatError {
    GrammarUnsupported,
    /// A recognized OpenAI response_format type that this server does not implement yet (e.g.
    /// `json_schema`). Distinct from `InvalidResponseFormat`: the request is well-formed.
    UnsupportedResponseFormat(&'static str),
    InvalidResponseFormat(String),
}

impl ResponseFormatError {
    fn message(&self) -> String {
        match self {
            ResponseFormatError::GrammarUnsupported => {
                "response_format with JSON constraints requires building mirai server with capability-grammar"
                    .to_string()
            },
            ResponseFormatError::UnsupportedResponseFormat(kind) => {
                format!("response_format type `{kind}` is recognized but not supported yet")
            },
            ResponseFormatError::InvalidResponseFormat(detail) => {
                format!("response_format is not a recognized object: {detail}")
            },
        }
    }

    fn code(&self) -> &'static str {
        match self {
            ResponseFormatError::GrammarUnsupported => "unsupported_response_format",
            ResponseFormatError::UnsupportedResponseFormat(_) => "unsupported_response_format",
            ResponseFormatError::InvalidResponseFormat(_) => "invalid_response_format",
        }
    }
}

fn request_error_response(error: ResponseFormatError) -> ChatCompletionResult {
    ChatCompletionResult::Error(status::Custom(
        Status::BadRequest,
        Json(OaiErrorResponse {
            error: OaiError {
                message: error.message(),
                kind: "invalid_request_error".to_string(),
                param: Some("response_format".to_string()),
                code: Some(error.code().to_string()),
            },
        }),
    ))
}

fn with_response_format_grammar(
    config: ChatReplyConfig,
    grammar: Grammar,
) -> Result<ChatReplyConfig, ResponseFormatError> {
    if !cfg!(feature = "capability-grammar") {
        return Err(ResponseFormatError::GrammarUnsupported);
    }

    Ok(config.with_grammar(Some(grammar)))
}

fn build_reply_config(request: &ChatCompletionRequest) -> Result<ChatReplyConfig, ResponseFormatError> {
    let token_limit = request.max_completion_tokens.or(request.max_tokens);
    let mut config = ChatReplyConfig::default().with_token_limit(token_limit);

    if request.temperature.is_some_and(|temperature| temperature <= 0.0) {
        config = config.with_sampling_method(SamplingMethod::Greedy {});
    } else if request.temperature.is_some() || request.top_p.is_some() || request.top_k.is_some() {
        config = config.with_sampling_method(SamplingMethod::Stochastic {
            temperature: request.temperature,
            top_k: request.top_k,
            top_p: request.top_p,
            min_p: None,
        });
    }

    // Interpret the raw value as a typed response_format here (rather than at extraction) so an
    // unrecognized object surfaces as our 400, not Rocket's 422. Only `text` and `json_object` are
    // supported. The `json_schema` form is recognized but rejected as unsupported (not malformed):
    // xgrammar silently ignores schema keywords it cannot enforce (multipleOf, not, external $ref,
    // multi-entry allOf), so accepting it as-is would violate OpenAI's strict guarantee. Enforcing
    // it correctly (reject those keywords up front, thread `strict` through `shoji::Grammar`) is
    // left to a follow-up.
    let response_format = match &request.response_format {
        Some(value) => {
            if value.get("type").and_then(serde_json::Value::as_str) == Some("json_schema") {
                return Err(ResponseFormatError::UnsupportedResponseFormat("json_schema"));
            }
            Some(
                serde_json::from_value::<ResponseFormat>(value.clone())
                    .map_err(|error| ResponseFormatError::InvalidResponseFormat(error.to_string()))?,
            )
        },
        None => None,
    };

    config = match response_format {
        // Map json_object to xgrammar's builtin JSON grammar, whose root is a JSON object or array
        // (it does not accept scalar roots like `42` or `"text"`), so a root array is allowed and
        // not just an object.
        //
        // Completeness contract: the grammar masks the stop token until the value is complete, so a
        // response that ends naturally (finish_reason=stop) is guaranteed to be complete, valid
        // JSON; a response truncated by the token limit (finish_reason=length) may be partial — the
        // client must check finish_reason, exactly as with OpenAI.
        //
        // Deliberate divergence from OpenAI: we do not require the messages to mention "json".
        // OpenAI uses that prompt guard to steer unconstrained sampling; here the grammar already
        // governs validity, so the guard is a prompt-quality nicety, not a correctness requirement,
        // and omitting it avoids rejecting otherwise-valid requests.
        Some(ResponseFormat::JsonObject) => with_response_format_grammar(config, Grammar::JsonAny {})?,
        Some(ResponseFormat::Text) | None => config,
    };

    Ok(config)
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

    let config = match build_reply_config(&request) {
        Ok(config) => config,
        Err(error) => return request_error_response(error),
    };
    let messages = to_chat_messages(&request.messages);

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
    use super::*;

    fn request(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).expect("valid request json")
    }

    fn reply_config(json: &str) -> ChatReplyConfig {
        build_reply_config(&request(json)).expect("valid reply config")
    }

    #[cfg(not(feature = "capability-grammar"))]
    fn reply_config_error(json: &str) -> ResponseFormatError {
        build_reply_config(&request(json)).expect_err("invalid reply config")
    }

    #[test]
    fn response_format_maps_to_grammar() {
        assert!(reply_config(r#"{"messages":[]}"#).grammar.is_none());
        assert!(reply_config(r#"{"messages":[],"response_format":{"type":"text"}}"#).grammar.is_none());

        #[cfg(feature = "capability-grammar")]
        assert_eq!(
            reply_config(r#"{"messages":[],"response_format":{"type":"json_object"}}"#).grammar,
            Some(Grammar::JsonAny {})
        );
    }

    #[test]
    fn response_format_rejects_grammar_without_capability() {
        #[cfg(not(feature = "capability-grammar"))]
        assert_eq!(
            reply_config_error(r#"{"messages":[],"response_format":{"type":"json_object"}}"#),
            ResponseFormatError::GrammarUnsupported
        );
    }

    #[test]
    fn response_format_unrecognized_is_invalid() {
        // An object that is not a recognized response_format type must surface as our request error
        // (400 invalid_response_format), not as Rocket's request-extraction failure (422). This
        // holds regardless of capability-grammar, since the value is interpreted before gating.
        let error = build_reply_config(&request(r#"{"messages":[],"response_format":{"type":"totally-bogus"}}"#))
            .expect_err("unrecognized response_format should be rejected");
        assert!(
            matches!(error, ResponseFormatError::InvalidResponseFormat(_)),
            "expected InvalidResponseFormat, got {error:?}"
        );
    }

    #[test]
    fn response_format_json_schema_is_unsupported_not_invalid() {
        // json_schema is a recognized OpenAI form, so it must be reported as unsupported (not as a
        // malformed/unrecognized object), rather than silently under-enforced. Holds regardless of
        // capability-grammar, since it is detected before gating.
        let error = build_reply_config(&request(
            r#"{"messages":[],"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}}"#,
        ))
        .expect_err("json_schema should be rejected as unsupported");
        assert_eq!(error, ResponseFormatError::UnsupportedResponseFormat("json_schema"));
        assert_eq!(error.code(), "unsupported_response_format");
    }

    #[test]
    fn response_format_validation_errors_are_request_errors() {
        match request_error_response(ResponseFormatError::GrammarUnsupported) {
            ChatCompletionResult::Error(_) => {},
            ChatCompletionResult::Json(_) | ChatCompletionResult::Stream(_) => {
                panic!("response_format validation errors should be request errors")
            },
        }
    }

    #[test]
    fn malformed_response_format_passes_json_extraction() {
        // The raw-value field is what keeps a malformed response_format out of Rocket's `Json` data
        // guard (which would answer 422). It must deserialize so the request reaches our handler,
        // where it is turned into a 400 instead.
        for body in [
            r#"{"messages":[],"response_format":{"type":"totally-bogus"}}"#,
            r#"{"messages":[],"response_format":{"type":"json_schema","json_schema":{"schema":{}}}}"#,
            r#"{"messages":[],"response_format":"not-even-an-object"}"#,
        ] {
            serde_json::from_str::<ChatCompletionRequest>(body)
                .unwrap_or_else(|error| panic!("expected {body} to pass extraction, got {error}"));
        }
    }

    // Test-only route returning a response_format validation error, used to exercise the actual
    // Rocket response layer. Defined at module level so the `rocket::get` macro stays local.
    #[rocket::get("/err")]
    fn err_route() -> ChatCompletionResult {
        request_error_response(ResponseFormatError::InvalidResponseFormat("bad".to_string()))
    }

    #[test]
    fn error_responder_yields_http_400_with_openai_body() {
        // A response_format validation error must come back as HTTP 400 with an OpenAI-shaped error
        // body, not Rocket's default 422/500 pages.
        let client = rocket::local::blocking::Client::tracked(rocket::build().mount("/", rocket::routes![err_route]))
            .expect("rocket client");
        let response = client.get("/err").dispatch();

        assert_eq!(response.status(), Status::BadRequest);
        let body: serde_json::Value = response.into_json().expect("json error body");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "response_format");
        assert_eq!(body["error"]["code"], "invalid_response_format");
        assert!(
            body["error"]["message"].as_str().is_some_and(|message| !message.is_empty()),
            "expected a non-empty error message, got {body}"
        );
    }

    #[cfg(feature = "capability-grammar")]
    #[test]
    fn response_format_composes_with_sampling_options() {
        let stochastic = reply_config(
            r#"{"messages":[],"temperature":0.7,"top_p":0.9,"top_k":40,"response_format":{"type":"json_object"}}"#,
        );
        assert_eq!(stochastic.grammar, Some(Grammar::JsonAny {}));
        assert_eq!(
            stochastic.sampling_policy,
            uzu::types::basic::SamplingPolicy::Custom {
                method: SamplingMethod::Stochastic {
                    temperature: Some(0.7),
                    top_k: Some(40),
                    top_p: Some(0.9),
                    min_p: None,
                },
            }
        );

        let greedy = reply_config(r#"{"messages":[],"temperature":0,"response_format":{"type":"json_object"}}"#);
        assert_eq!(greedy.grammar, Some(Grammar::JsonAny {}));
        assert_eq!(
            greedy.sampling_policy,
            uzu::types::basic::SamplingPolicy::Custom {
                method: SamplingMethod::Greedy {},
            }
        );
    }
}
