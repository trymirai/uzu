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
        basic::{Grammar, SamplingMethod, ToolCall, ToolDescription, ToolFunction, ToolNamespace, Value},
        session::chat::{ChatMessage, ChatReplyConfig, ChatReplyFinishReason, ChatReplyStats, ChatRole},
    },
};

use crate::server::ServerState;

#[derive(Serialize, Deserialize, Clone)]
pub struct OaiMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OaiToolCall>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OaiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OaiFunctionCall,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OaiFunctionCall {
    pub name: String,
    pub arguments: String,
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
    // Raw value (not typed) so a bad response_format is our 400, not Rocket's 422.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<OaiTool>>,
    // Raw value (not typed) so an unsupported tool_choice is our 400, not Rocket's 422.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct OaiTool {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub kind: String,
    pub function: OaiToolFunction,
}

#[derive(Deserialize)]
pub struct OaiToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
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

fn to_tool_namespaces(tools: &[OaiTool]) -> Vec<ToolNamespace> {
    if tools.is_empty() {
        return Vec::new();
    }
    let descriptions = tools
        .iter()
        .map(|tool| ToolDescription::Function {
            tool_function: ToolFunction {
                name: tool.function.name.clone(),
                description: tool.function.description.clone().unwrap_or_default(),
                parameters: tool.function.parameters.clone().map(Value::from),
                return_definition: None,
            },
        })
        .collect();
    vec![ToolNamespace {
        name: "functions".to_string(),
        description: None,
        tools: descriptions,
    }]
}

fn to_oai_tool_calls(tool_calls: &[ToolCall]) -> Option<Vec<OaiToolCall>> {
    if tool_calls.is_empty() {
        return None;
    }
    Some(
        tool_calls
            .iter()
            .map(|tool_call| OaiToolCall {
                id: tool_call.identifier.clone().unwrap_or_else(|| format!("call_{}", Uuid::new_v4().simple())),
                kind: "function".to_string(),
                function: OaiFunctionCall {
                    name: tool_call.name.clone(),
                    arguments: serde_json::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_string()),
                },
            })
            .collect(),
    )
}

fn to_chat_messages(
    messages: &[OaiMessage],
    tool_namespaces: Vec<ToolNamespace>,
) -> Vec<ChatMessage> {
    let mut chat_messages: Vec<ChatMessage> = messages
        .iter()
        .map(|message| {
            let role = ChatRole::from_str(&message.role).unwrap_or(ChatRole::User {});
            ChatMessage::for_role(role).with_text(message.content.clone().unwrap_or_default())
        })
        .collect();
    if !tool_namespaces.is_empty()
        && let Some(first) = chat_messages.first_mut()
    {
        *first = first.clone().with_tool_namespaces(tool_namespaces);
    }
    chat_messages
}

#[derive(Debug, PartialEq, Eq)]
enum ResponseFormatError {
    GrammarUnsupported,
    InvalidResponseFormat(String),
    InvalidJsonSchema(String),
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

fn bad_request(
    message: String,
    param: &str,
    code: &str,
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

fn request_error_response(error: ResponseFormatError) -> ChatCompletionResult {
    bad_request(error.message(), "response_format", error.code())
}

#[derive(Debug, PartialEq, Eq)]
enum ToolRequestError {
    StreamingUnsupported,
    UnsupportedToolChoice,
}

impl ToolRequestError {
    fn message(&self) -> &'static str {
        match self {
            ToolRequestError::StreamingUnsupported => "tool calling is not supported with stream: true yet",
            ToolRequestError::UnsupportedToolChoice => "tool_choice values other than \"auto\" are not supported yet",
        }
    }

    fn param(&self) -> &'static str {
        match self {
            ToolRequestError::StreamingUnsupported => "stream",
            ToolRequestError::UnsupportedToolChoice => "tool_choice",
        }
    }

    fn code(&self) -> &'static str {
        match self {
            ToolRequestError::StreamingUnsupported => "unsupported_streaming_tools",
            ToolRequestError::UnsupportedToolChoice => "unsupported_tool_choice",
        }
    }
}

// Tools only support the default `auto` choice, non-streaming, for now; reject the rest
// with a 400 instead of silently emitting tool calls the response path cannot deliver.
fn ensure_tools_supported(request: &ChatCompletionRequest) -> Result<(), ToolRequestError> {
    let has_tools = request.tools.as_ref().is_some_and(|tools| !tools.is_empty());
    if !has_tools {
        return Ok(());
    }
    if let Some(choice) = &request.tool_choice
        && choice.as_str() != Some("auto")
    {
        return Err(ToolRequestError::UnsupportedToolChoice);
    }
    if request.stream.unwrap_or(false) {
        return Err(ToolRequestError::StreamingUnsupported);
    }
    Ok(())
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
                role: "assistant".to_string(),
                content: Some(format!("Error: {message}")),
                tool_calls: None,
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
            Some(reply) => {
                let tool_calls = to_oai_tool_calls(&reply.message.tool_calls());
                let text = reply.message.text().unwrap_or_default();
                let content = if text.is_empty() && tool_calls.is_some() {
                    None
                } else {
                    Some(text)
                };
                ChatCompletionResponse {
                    id,
                    object: "chat.completion".to_string(),
                    created,
                    model,
                    choices: vec![ChatCompletionChoice {
                        index: 0,
                        message: OaiMessage {
                            role: "assistant".to_string(),
                            content,
                            tool_calls,
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
    if let Err(error) = ensure_tools_supported(&request) {
        return bad_request(error.message().to_string(), error.param(), error.code());
    }
    let tool_namespaces = to_tool_namespaces(request.tools.as_deref().unwrap_or(&[]));
    let messages = to_chat_messages(&request.messages, tool_namespaces);

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
#[path = "../../unit/server/chat_completions_test.rs"]
mod tests;
