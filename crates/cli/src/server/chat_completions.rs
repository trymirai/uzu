use std::{
    fmt,
    pin::Pin,
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
    session::chat::{ChatSession, ChatSessionStream, ChatSessionStreamChunk},
    types::{
        basic::{Grammar, SamplingMethod},
        session::chat::{ChatMessage, ChatReplyConfig, ChatReplyFinishReason, ChatReplyStats},
    },
};

use crate::server::{
    ServerState,
    chat_tool_calls::{
        OaiTool, OaiToolCall, StreamToolCall, normalize_tool_calls_for_capability, response_tool_calls, select_tools,
        stream_tool_call_deltas, to_chat_messages, tool_call_batch_error, validate_parallel_tool_calls,
        validate_selected_tools, validate_tool_capability, validate_tool_history,
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OaiRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
    Unsupported(String),
}

impl OaiRole {
    fn as_str(&self) -> &str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Unsupported(role) => role,
        }
    }
}

impl From<String> for OaiRole {
    fn from(role: String) -> Self {
        if role == Self::System.as_str() {
            Self::System
        } else if role == Self::Developer.as_str() {
            Self::Developer
        } else if role == Self::User.as_str() {
            Self::User
        } else if role == Self::Assistant.as_str() {
            Self::Assistant
        } else if role == Self::Tool.as_str() {
            Self::Tool
        } else {
            Self::Unsupported(role)
        }
    }
}

impl fmt::Display for OaiRole {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for OaiRole {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for OaiRole {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer).map(Self::from)
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OaiMessage {
    pub role: OaiRole,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OaiToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<OaiMessage>,
    #[serde(default)]
    pub tools: Option<Vec<OaiTool>>,
    // Keep this raw so invalid or unsupported choices produce our OpenAI-style 400 response.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
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
    // Raw value (not typed) so a bad response_format is our 400, not Rocket's 422.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
}

impl ChatCompletionRequest {
    fn parse(value: serde_json::Value) -> Result<Self, RequestBodyError> {
        serde_json::from_value(value).map_err(|error| RequestBodyError {
            message: format!("request body does not match the Chat Completions schema: {error}"),
        })
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        json_schema: JsonSchemaFormat,
    },
}

#[derive(Deserialize)]
pub struct JsonSchemaFormat {
    pub schema: serde_json::Value,
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
    role: Option<OaiRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<StreamToolCall>>,
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

#[derive(Debug, PartialEq, Eq)]
enum ResponseFormatError {
    GrammarUnsupported,
    InvalidResponseFormat(String),
    InvalidJsonSchema(String),
}

#[derive(Debug, PartialEq, Eq)]
struct RequestBodyError {
    message: String,
}

impl<'a> From<rocket::serde::json::Error<'a>> for RequestBodyError {
    fn from(error: rocket::serde::json::Error<'a>) -> Self {
        Self {
            message: format!("request body is not valid JSON: {error}"),
        }
    }
}

impl ResponseFormatError {
    fn message(&self) -> String {
        match self {
            ResponseFormatError::GrammarUnsupported => {
                "response_format with JSON constraints requires building mirai server with capability-grammar"
                    .to_string()
            },
            ResponseFormatError::InvalidResponseFormat(detail) => {
                format!("response_format is not a recognized object: {detail}")
            },
            ResponseFormatError::InvalidJsonSchema(detail) => {
                format!("response_format.json_schema.schema is not a valid JSON Schema: {detail}")
            },
        }
    }

    fn code(&self) -> &'static str {
        match self {
            ResponseFormatError::GrammarUnsupported => "unsupported_response_format",
            ResponseFormatError::InvalidResponseFormat(_) => "invalid_response_format",
            ResponseFormatError::InvalidJsonSchema(_) => "invalid_json_schema",
        }
    }
}

fn request_error_response(error: ResponseFormatError) -> ChatCompletionResult {
    invalid_request_response("response_format", error.code(), error.message())
}

pub(super) fn invalid_request_response(
    param: &str,
    code: &str,
    message: String,
) -> ChatCompletionResult {
    ChatCompletionResult::Error(status::Custom(
        Status::BadRequest,
        Json(OaiErrorResponse {
            error: OaiError {
                message,
                kind: "invalid_request_error".to_string(),
                param: Some(param.to_string()),
                code: Some(code.to_string()),
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

fn json_schema_grammar(json_schema: &JsonSchemaFormat) -> Result<Grammar, ResponseFormatError> {
    jsonschema::meta::validate(&json_schema.schema)
        .map_err(|error| ResponseFormatError::InvalidJsonSchema(error.to_string()))?;
    let schema = serde_json::to_string(&json_schema.schema)
        .map_err(|error| ResponseFormatError::InvalidResponseFormat(error.to_string()))?;
    Ok(Grammar::JsonSchema {
        schema,
    })
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
            repetition_penalty: None,
            suffix_repetition_length: None,
        });
    }

    let response_format = match &request.response_format {
        Some(value) => Some(
            serde_json::from_value::<ResponseFormat>(value.clone())
                .map_err(|error| ResponseFormatError::InvalidResponseFormat(error.to_string()))?,
        ),
        None => None,
    };

    config = match response_format {
        Some(ResponseFormat::JsonObject) => with_response_format_grammar(config, Grammar::JsonAny {})?,
        Some(ResponseFormat::JsonSchema {
            json_schema,
        }) => with_response_format_grammar(config, json_schema_grammar(&json_schema)?)?,
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
                role: OaiRole::Assistant,
                content: Some(format!("Error: {message}")),
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: ChatCompletionUsage::default(),
    }
}

fn error_response_with_text(
    id: String,
    model: String,
    created: i64,
    text: Option<String>,
    error: &str,
) -> ChatCompletionResponse {
    let mut response = error_response(id, model, created, error);
    if let Some(text) = text.filter(|text| !text.is_empty()) {
        response.choices[0].message.content = Some(format!("{text}\n\nError: {error}"));
    }
    response
}

fn stream_error_text(
    error: &str,
    has_emitted_text: bool,
) -> String {
    format!(
        "{}Error: {error}",
        if has_emitted_text {
            "\n\n"
        } else {
            ""
        }
    )
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

fn next_text_delta(
    text: &str,
    emitted: &mut usize,
) -> Option<String> {
    let start = (*emitted..=text.len()).find(|&index| text.is_char_boundary(index)).unwrap_or(text.len());
    (text.len() > start).then(|| {
        *emitted = text.len();
        text[start..].to_string()
    })
}

async fn cancel_and_drain_stream(stream: &ChatSessionStream) {
    stream.cancel_token().cancel();
    while stream.next().await.is_some() {}
}

async fn run_blocking(
    session: Arc<Mutex<ChatSession>>,
    messages: Vec<ChatMessage>,
    config: ChatReplyConfig,
    id: String,
    model: String,
    created: i64,
    supports_multiple_tool_calls: bool,
    allowed_tool_names: Vec<String>,
) -> ChatCompletionResponse {
    let session = session.lock().await;
    if let Err(error) = session.reset().await {
        return error_response(id, model, created, &error.to_string());
    }

    match session.reply(messages, config).await {
        Ok(replies) => match replies.last() {
            Some(reply) => {
                let message = normalize_tool_calls_for_capability(&reply.message, supports_multiple_tool_calls);
                if let Some(error) =
                    tool_call_batch_error(&message, reply.finish_reason.as_ref(), &allowed_tool_names, true)
                {
                    return error_response_with_text(id, model, created, message.text(), &error);
                }
                ChatCompletionResponse {
                    id,
                    object: "chat.completion".to_string(),
                    created,
                    model,
                    choices: vec![ChatCompletionChoice {
                        index: 0,
                        message: OaiMessage {
                            role: OaiRole::Assistant,
                            content: message.text(),
                            tool_calls: response_tool_calls(&message),
                            tool_call_id: None,
                        },
                        finish_reason: reply
                            .finish_reason
                            .as_ref()
                            .map(map_finish_reason)
                            .unwrap_or_else(|| "stop".to_string()),
                    }],
                    usage: usage_from_stats(&reply.stats),
                }
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
    supports_multiple_tool_calls: bool,
    allowed_tool_names: Vec<String>,
    sender: mpsc::UnboundedSender<Event>,
) {
    let session = tokio::select! {
        session = session.lock() => session,
        _ = sender.closed() => return,
    };
    if let Err(error) = session.reset().await {
        let _ = sender.send(Event::data(chunk_json(
            &id,
            &model,
            created,
            StreamDelta {
                role: Some(OaiRole::Assistant),
                content: Some(format!("Error: {error}")),
                tool_calls: None,
            },
            Some("stop".to_string()),
            None,
        )));
        let _ = sender.send(Event::data("[DONE]"));
        return;
    }

    if sender
        .send(Event::data(chunk_json(
            &id,
            &model,
            created,
            StreamDelta {
                role: Some(OaiRole::Assistant),
                content: None,
                tool_calls: None,
            },
            None,
            None,
        )))
        .is_err()
    {
        return;
    }

    let stream = tokio::select! {
        stream = session.reply_with_stream(messages, config) => stream,
        _ = sender.closed() => return,
    };
    let mut emitted = 0usize;
    let mut emitted_tool_calls = 0usize;
    let mut finish_reason = "stop".to_string();
    let mut usage = ChatCompletionUsage::default();
    let mut errored = false;
    let mut tool_call_error: Option<String> = None;
    let mut last_message: Option<ChatMessage> = None;
    let mut last_finish_reason: Option<ChatReplyFinishReason> = None;

    loop {
        let chunk = tokio::select! {
            chunk = stream.next() => chunk,
            _ = sender.closed() => {
                cancel_and_drain_stream(&stream).await;
                return;
            },
        };
        let Some(chunk) = chunk else {
            break;
        };
        match chunk {
            ChatSessionStreamChunk::Replies {
                replies,
            } => {
                let Some(reply) = replies.last() else {
                    continue;
                };
                let is_terminal = reply.finish_reason.is_some();
                let message = normalize_tool_calls_for_capability(&reply.message, supports_multiple_tool_calls);
                tool_call_error =
                    tool_call_batch_error(&message, reply.finish_reason.as_ref(), &allowed_tool_names, is_terminal);
                last_message = Some(message.clone());
                last_finish_reason = reply.finish_reason.clone();
                let text = message.text().unwrap_or_default();
                if let Some(delta) = next_text_delta(&text, &mut emitted) {
                    let sent = sender.send(Event::data(chunk_json(
                        &id,
                        &model,
                        created,
                        StreamDelta {
                            role: None,
                            content: Some(delta),
                            tool_calls: None,
                        },
                        None,
                        None,
                    )));
                    if sent.is_err() {
                        cancel_and_drain_stream(&stream).await;
                        return;
                    }
                }
                if let Some(reason) = &reply.finish_reason {
                    if let Some(error) = &tool_call_error {
                        errored = true;
                        let _ = sender.send(Event::data(chunk_json(
                            &id,
                            &model,
                            created,
                            StreamDelta {
                                role: None,
                                content: Some(stream_error_text(error, emitted > 0)),
                                tool_calls: None,
                            },
                            Some("stop".to_string()),
                            None,
                        )));
                        cancel_and_drain_stream(&stream).await;
                        break;
                    }
                    let tool_calls = message.tool_calls();
                    let tool_call_deltas = stream_tool_call_deltas(&tool_calls, &mut emitted_tool_calls);
                    if !tool_call_deltas.is_empty() {
                        let sent = sender.send(Event::data(chunk_json(
                            &id,
                            &model,
                            created,
                            StreamDelta {
                                role: None,
                                content: None,
                                tool_calls: Some(tool_call_deltas),
                            },
                            None,
                            None,
                        )));
                        if sent.is_err() {
                            cancel_and_drain_stream(&stream).await;
                            return;
                        }
                    }
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
                        content: Some(stream_error_text(&error.to_string(), emitted > 0)),
                        tool_calls: None,
                    },
                    Some("stop".to_string()),
                    None,
                )));
                cancel_and_drain_stream(&stream).await;
                break;
            },
        }
    }

    if !errored && last_message.is_none() {
        errored = true;
        let _ = sender.send(Event::data(chunk_json(
            &id,
            &model,
            created,
            StreamDelta {
                role: None,
                content: Some("Error: No response generated".to_string()),
                tool_calls: None,
            },
            Some("stop".to_string()),
            None,
        )));
    }

    if !errored && let Some(message) = &last_message {
        tool_call_error = tool_call_batch_error(message, last_finish_reason.as_ref(), &allowed_tool_names, true);
    }

    if !errored && let Some(error) = &tool_call_error {
        errored = true;
        let _ = sender.send(Event::data(chunk_json(
            &id,
            &model,
            created,
            StreamDelta {
                role: None,
                content: Some(stream_error_text(error, emitted > 0)),
                tool_calls: None,
            },
            Some("stop".to_string()),
            None,
        )));
    }

    if !errored {
        let _ = sender.send(Event::data(chunk_json(
            &id,
            &model,
            created,
            StreamDelta {
                role: None,
                content: None,
                tool_calls: None,
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
    request: Result<Json<serde_json::Value>, rocket::serde::json::Error<'_>>,
    state: &State<ServerState>,
) -> ChatCompletionResult {
    let request = match request {
        Ok(request) => request.into_inner(),
        Err(error) => {
            return invalid_request_response("request", "invalid_request", RequestBodyError::from(error).message);
        },
    };
    let request = match ChatCompletionRequest::parse(request) {
        Ok(request) => request,
        Err(error) => return invalid_request_response("request", "invalid_request", error.message),
    };
    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let created = now_unix();
    let model = state.model_name.clone();
    let is_stream = request.stream.unwrap_or(false);

    let config = match build_reply_config(&request) {
        Ok(config) => config,
        Err(error) => return request_error_response(error),
    };
    let tools = match select_tools(&request) {
        Ok(tools) => tools,
        Err(error) => return invalid_request_response("tool_choice", error.code(), error.message()),
    };
    if let Err(error) = validate_selected_tools(&request, &tools) {
        return invalid_request_response(&error.param(), error.code(), error.message());
    }
    let (supports_tool_calls, supports_multiple_tool_calls) = {
        let session = state.session.lock().await;
        (session.supports_tool_calls().await, session.supports_multiple_tool_calls().await)
    };
    if let Err(error) = validate_tool_capability(&tools, supports_tool_calls) {
        return invalid_request_response(&error.param(), error.code(), error.message());
    }
    if let Err(error) = validate_tool_history(&request.messages, supports_tool_calls, supports_multiple_tool_calls) {
        return invalid_request_response(&error.param(), error.code(), error.message());
    }
    if let Err(error) = validate_parallel_tool_calls(&request, &tools, supports_multiple_tool_calls) {
        return invalid_request_response("parallel_tool_calls", error.code(), error.message());
    }
    let allowed_tool_names = tools.iter().map(|tool| tool.function.name.clone()).collect::<Vec<_>>();
    let messages = to_chat_messages(&request.messages, tools);

    if is_stream {
        let session = Arc::clone(&state.session);
        let (sender, receiver) = mpsc::unbounded_channel::<Event>();
        rocket::tokio::spawn(run_stream(
            session,
            messages,
            config,
            id,
            model,
            created,
            supports_multiple_tool_calls,
            allowed_tool_names,
            sender,
        ));
        let body: Pin<Box<dyn Stream<Item = Event> + Send>> = Box::pin(UnboundedReceiverStream::new(receiver));
        ChatCompletionResult::Stream(EventStream::from(body))
    } else {
        let session = Arc::clone(&state.session);
        let response = run_blocking(
            session,
            messages,
            config,
            id,
            model,
            created,
            supports_multiple_tool_calls,
            allowed_tool_names,
        )
        .await;
        ChatCompletionResult::Json(Json(response))
    }
}

#[cfg(test)]
#[path = "../../unit/server/chat_completions_test.rs"]
mod tests;
