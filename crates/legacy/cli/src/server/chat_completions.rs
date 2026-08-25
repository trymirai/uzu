use std::{
    collections::HashMap,
    pin::Pin,
    str::FromStr,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use rocket::{
    Request, State,
    data::{ByteUnit, Data},
    futures::Stream,
    http::{ContentType, Status},
    post,
    response::{
        Responder, status,
        stream::{ByteStream, Event, EventStream},
    },
    serde::json::Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;
use uzu::{
    session::chat::{ChatSession, ChatSessionStream, ChatSessionStreamChunk, UNPARSED_ARGUMENTS_KEY},
    types::{
        basic::{Grammar, ReasoningEffort, SamplingMethod},
        session::chat::{
            ChatContentBlock, ChatMessage, ChatReplyConfig, ChatReplyFinishReason, ChatReplyStats, ChatRole,
        },
    },
};

use crate::{
    common::model_capabilities::ThinkingSupport,
    server::{
        ServerState,
        chat_tool_calls::{
            OaiTool, OaiToolCall, ToolCallStreamer, backfill_tool_result_names, choose_tools, insert_tools_message,
            oai_tool_call, reply_tool_calls, to_tool_call, tool_call_result_block, withhold_stream_text,
        },
        request_log::RequestLog,
    },
};

fn deserialize_message_content<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match Option::<serde_json::Value>::deserialize(deserializer)? {
        None => Ok(None),
        Some(serde_json::Value::String(text)) => Ok(Some(text)),
        Some(serde_json::Value::Array(parts)) => {
            let mut text = String::new();
            for part in &parts {
                match part.get("type").and_then(serde_json::Value::as_str) {
                    Some("text") => {
                        text.push_str(part.get("text").and_then(serde_json::Value::as_str).unwrap_or_default());
                    },
                    Some(other) => {
                        return Err(serde::de::Error::custom(format!(
                            "unsupported message content part type: {other}"
                        )));
                    },
                    None => return Err(serde::de::Error::custom("message content part is missing a type")),
                }
            }
            Ok(Some(text))
        },
        Some(_) => Err(serde::de::Error::custom("message content must be a string or an array of content parts")),
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OaiMessage {
    pub role: String,
    #[serde(default, deserialize_with = "deserialize_message_content")]
    pub content: Option<String>,
    // vLLM-style reasoning channel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OaiToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
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
    pub tools: Option<Vec<OaiTool>>,
    // Raw value like response_format: a bad tool_choice is our 400, not Rocket's 422.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    // Raw value like response_format: a bad reasoning_effort is our 400, not Rocket's 422.
    #[serde(default)]
    pub reasoning_effort: Option<serde_json::Value>,
    // vLLM-style thinking switch. Raw values like reasoning_effort: malformed ones are our 400.
    #[serde(default)]
    pub enable_thinking: Option<serde_json::Value>,
    // vLLM-style container; only its enable_thinking key is honored, other keys are ignored.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
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

#[derive(Serialize, Clone)]
pub struct ChatCompletionPromptTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Serialize, Clone, Default)]
pub struct ChatCompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<ChatCompletionPromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_verify_ct: Option<u32>,
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

#[derive(Serialize, Default)]
struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OaiToolCall>>,
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
    // The single JSON body arrives as a stream: Rocket only surfaces a client
    // disconnect by dropping the body stream, which is the blocking path's one
    // signal to cancel a generation nobody is waiting for.
    Json(ByteStream<Pin<Box<dyn Stream<Item = Vec<u8>> + Send>>>),
    Stream(EventStream<Pin<Box<dyn Stream<Item = Event> + Send>>>),
    Error(status::Custom<Json<OaiErrorResponse>>),
}

impl<'r> Responder<'r, 'r> for ChatCompletionResult {
    fn respond_to(
        self,
        request: &'r Request<'_>,
    ) -> rocket::response::Result<'r> {
        match self {
            ChatCompletionResult::Json(body) => (ContentType::JSON, body).respond_to(request),
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
            let mut chat_message = ChatMessage::for_role(role);
            // Block order matches how the session stores generated replies
            // (reasoning, then text, then tool calls) so a client that echoes
            // reasoning back keeps the message prefix intact.
            if let Some(reasoning) = message.reasoning_content.as_ref().filter(|reasoning| !reasoning.is_empty()) {
                chat_message = chat_message.with_reasoning(reasoning.clone());
            }
            if let Some(identifier) = &message.tool_call_id {
                let result = tool_call_result_block(identifier, message.content.clone().unwrap_or_default());
                chat_message = chat_message.with_block(result);
            } else if let Some(content) = &message.content {
                chat_message = chat_message.with_text(content.clone());
            }
            for tool_call in message.tool_calls.iter().flatten() {
                chat_message = chat_message.with_tool_call(to_tool_call(tool_call));
            }
            chat_message
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum MessageBuildError {
    ToolChoice(String),
    ReasoningEffort(String),
    EnableThinking(String),
    ChatTemplateKwargs(String),
}

impl MessageBuildError {
    fn param(&self) -> &'static str {
        match self {
            Self::ToolChoice(_) => "tool_choice",
            Self::ReasoningEffort(_) => "reasoning_effort",
            Self::EnableThinking(_) => "enable_thinking",
            Self::ChatTemplateKwargs(_) => "chat_template_kwargs",
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::ToolChoice(_) => "invalid_tool_choice",
            Self::ReasoningEffort(_) => "invalid_reasoning_effort",
            Self::EnableThinking(_) => "invalid_enable_thinking",
            Self::ChatTemplateKwargs(_) => "invalid_chat_template_kwargs",
        }
    }

    fn into_detail(self) -> String {
        match self {
            Self::ToolChoice(detail)
            | Self::ReasoningEffort(detail)
            | Self::EnableThinking(detail)
            | Self::ChatTemplateKwargs(detail) => detail,
        }
    }
}

fn parse_reasoning_effort(request: &ChatCompletionRequest) -> Result<Option<ReasoningEffort>, String> {
    let Some(value) = &request.reasoning_effort else {
        return Ok(None);
    };
    let raw = value.as_str().ok_or_else(|| "reasoning_effort must be a string".to_string())?;
    ReasoningEffort::from_str(raw).map(Some)
}

fn parse_enable_thinking(request: &ChatCompletionRequest) -> Result<Option<bool>, MessageBuildError> {
    let top_level = request
        .enable_thinking
        .as_ref()
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| MessageBuildError::EnableThinking("enable_thinking must be a boolean".to_string()))
        })
        .transpose()?;
    let template_kwarg = request
        .chat_template_kwargs
        .as_ref()
        .map(|value| {
            let object = value.as_object().ok_or_else(|| {
                MessageBuildError::ChatTemplateKwargs("chat_template_kwargs must be an object".to_string())
            })?;
            object
                .get("enable_thinking")
                .filter(|value| !value.is_null())
                .map(|value| {
                    value.as_bool().ok_or_else(|| {
                        MessageBuildError::EnableThinking(
                            "chat_template_kwargs.enable_thinking must be a boolean".to_string(),
                        )
                    })
                })
                .transpose()
        })
        .transpose()?
        .flatten();
    match (top_level, template_kwarg) {
        (Some(top_level), Some(template_kwarg)) if top_level != template_kwarg => {
            Err(MessageBuildError::EnableThinking(
                "enable_thinking and chat_template_kwargs.enable_thinking disagree".to_string(),
            ))
        },
        (top_level, template_kwarg) => Ok(top_level.or(template_kwarg)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningSource {
    ReasoningEffort,
    EnableThinking,
}

impl ReasoningSource {
    fn error(
        self,
        detail: String,
    ) -> MessageBuildError {
        match self {
            Self::ReasoningEffort => MessageBuildError::ReasoningEffort(detail),
            Self::EnableThinking => MessageBuildError::EnableThinking(detail),
        }
    }
}

fn requested_reasoning_effort(
    request: &ChatCompletionRequest
) -> Result<Option<(ReasoningEffort, ReasoningSource)>, MessageBuildError> {
    let explicit = parse_reasoning_effort(request).map_err(MessageBuildError::ReasoningEffort)?;
    let enable = parse_enable_thinking(request)?;
    match (explicit, enable) {
        (Some(effort), Some(enable)) if (effort == ReasoningEffort::Disabled) == enable => {
            Err(MessageBuildError::ReasoningEffort(format!(
                "reasoning_effort {effort} conflicts with enable_thinking {enable}"
            )))
        },
        (Some(effort), _) => Ok(Some((effort, ReasoningSource::ReasoningEffort))),
        (None, Some(enable)) => {
            let toggled = if enable {
                ReasoningEffort::Default
            } else {
                ReasoningEffort::Disabled
            };
            Ok(Some((toggled, ReasoningSource::EnableThinking)))
        },
        (None, None) => Ok(None),
    }
}

pub(crate) fn build_messages(
    request: &ChatCompletionRequest,
    thinking_support: ThinkingSupport,
) -> Result<Vec<ChatMessage>, MessageBuildError> {
    let tools =
        choose_tools(request.tools.as_deref(), request.tool_choice.as_ref()).map_err(MessageBuildError::ToolChoice)?;
    let mut messages = to_chat_messages(&request.messages);
    if let Some((effort, source)) = requested_reasoning_effort(request)? {
        let fulfilled = thinking_support.fulfill_requested_effort(effort).map_err(|detail| source.error(detail))?;
        if let Some(effort) = fulfilled {
            // The engine reads the effort from a reasoning_effort block carried on a system
            // message. Merge into a leading system message when the client sent one:
            // templates reject two system messages in a row.
            match messages.first_mut() {
                Some(first) if first.role == (ChatRole::System {}) => {
                    *first = first.clone().with_reasoning_effort(effort);
                },
                _ => messages.insert(0, ChatMessage::system().with_reasoning_effort(effort)),
            }
        }
    }
    backfill_tool_result_names(&mut messages);
    insert_tools_message(&mut messages, &tools);
    Ok(messages)
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

fn invalid_request_response(
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

fn usage_from_stats(
    stats: &ChatReplyStats,
    prefix_cache: bool,
) -> ChatCompletionUsage {
    let cached_tokens = stats.tokens_count_input_cached.unwrap_or(0);
    let prefilled_tokens = stats.tokens_count_input.unwrap_or(0);
    let prompt_tokens = prefilled_tokens + cached_tokens;
    let completion_tokens = stats.tokens_count_output.unwrap_or(0);
    ChatCompletionUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        prompt_tokens_details: prefix_cache.then_some(ChatCompletionPromptTokensDetails {
            cached_tokens,
        }),
        spec_verify_ct: stats
            .speculator_stats
            .as_ref()
            .map(|speculator_stats| speculator_stats.num_decode_forward_passes),
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
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
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

/// Tool call identifiers are assigned per turn and never rendered into tokens, so they are
/// excluded from the comparison; everything else about the messages must match.
fn messages_have_prefix(
    messages: &[ChatMessage],
    prefix: &[ChatMessage],
) -> bool {
    prefix.len() <= messages.len()
        && prefix.iter().zip(messages).all(|(prefix_message, message)| {
            prefix_message.role == message.role
                && prefix_message.metadata == message.metadata
                && prefix_message.content.len() == message.content.len()
                && prefix_message.content.iter().zip(&message.content).all(|(prefix_block, block)| {
                    match (prefix_block, block) {
                        (
                            ChatContentBlock::ToolCall {
                                value: prefix_call,
                            },
                            ChatContentBlock::ToolCall {
                                value: call,
                            },
                        ) => {
                            let (mut prefix_call, mut call) = (prefix_call.clone(), call.clone());
                            prefix_call.identifier = None;
                            call.identifier = None;
                            prefix_call == call
                        },
                        _ => prefix_block == block,
                    }
                })
        })
}

/// Returns only the messages the session has not seen yet when the request
/// extends the session's current history; otherwise resets the session and
/// returns the full list. Token-level reuse itself is decided inside the
/// session, which falls back to a full prefill if the rendered tokens diverge.
async fn prepare_input(
    session: &ChatSession,
    mut messages: Vec<ChatMessage>,
    prefix_cache: bool,
) -> Result<Vec<ChatMessage>, uzu::session::chat::ChatSessionError> {
    if prefix_cache {
        let current = session.messages().await;
        if !current.is_empty() && messages.len() > current.len() && messages_have_prefix(&messages, &current) {
            return Ok(messages.split_off(current.len()));
        }
    }
    session.reset().await?;
    Ok(messages)
}

fn send_response(
    sender: &mpsc::UnboundedSender<Vec<u8>>,
    response: &ChatCompletionResponse,
) {
    let _ = sender.send(serde_json::to_vec(response).unwrap_or_default());
}

/// Cancels the turn and waits for it to wind down before the caller releases
/// the session lock: a request arriving right after a cancellation must not
/// catch the session mid-cleanup, where it rejects operations.
async fn cancel_and_drain(stream: &ChatSessionStream) {
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
    prefix_cache: bool,
    sender: mpsc::UnboundedSender<Vec<u8>>,
    log: RequestLog,
) {
    // A blocking response writes nothing until generation finishes, and Rocket
    // only discovers a vanished client through a failed socket write, which is
    // what makes it drop the body receiver that `sender.closed()` observes. A
    // periodic newline — insignificant leading whitespace for the JSON body —
    // keeps forcing that write while the request queues and generates.
    let mut keepalive = tokio::time::interval(std::time::Duration::from_secs(3));

    // Requests serialize on the single session, so a client can disconnect
    // while still queued; generating for it would block everyone behind it.
    let mut lock = std::pin::pin!(session.lock());
    let session = loop {
        tokio::select! {
            session = &mut lock => break session,
            () = sender.closed() => {
                log.fail("client disconnected");
                return;
            },
            _ = keepalive.tick() => {
                let _ = sender.send(b"\n".to_vec());
            },
        }
    };
    let input = match prepare_input(&session, messages, prefix_cache).await {
        Ok(input) => input,
        Err(error) => {
            log.fail(&error.to_string());
            send_response(&sender, &error_response(id, model, created, &error.to_string()));
            return;
        },
    };

    // Drive the reply through its stream instead of ChatSession::reply, which
    // cannot be cancelled: racing each chunk against the channel closing
    // notices a disconnect mid-generation. Cancelling stops the backend at the
    // next token boundary and lets the session run its cancelled-turn cleanup.
    let stream = session.reply_with_stream(input, config).await;
    let mut final_replies = Vec::new();
    loop {
        let chunk = tokio::select! {
            chunk = stream.next() => chunk,
            () = sender.closed() => {
                cancel_and_drain(&stream).await;
                log.fail("client disconnected");
                return;
            },
            _ = keepalive.tick() => {
                let _ = sender.send(b"\n".to_vec());
                continue;
            },
        };
        let Some(chunk) = chunk else {
            break;
        };
        match chunk {
            ChatSessionStreamChunk::Replies {
                replies,
            } => final_replies = replies,
            ChatSessionStreamChunk::Error {
                error,
            } => {
                log.fail(&error.to_string());
                send_response(&sender, &error_response(id, model, created, &error.to_string()));
                return;
            },
        }
    }

    let response = match final_replies.last() {
        Some(reply) => {
            let tool_calls = reply_tool_calls(&reply.message);
            // OpenAI sets content to null (not "") on tool call replies.
            let content = reply.message.text().or_else(|| tool_calls.is_none().then(String::new));
            let mut finish_reason =
                reply.finish_reason.as_ref().map(map_finish_reason).unwrap_or_else(|| "stop".to_string());
            // Calls the parser could not finalize must not execute as a partial batch.
            // OpenAI has no finish reason for this; providers signal terminal errors with
            // unrecognized reasons (pi maps them to a thrown provider error), so the turn
            // fails loudly instead of stalling on an absent or incomplete tool_calls array.
            let has_unfinished_candidates =
                reply.message.content.iter().any(|block| matches!(block, ChatContentBlock::ToolCallCandidate { .. }));
            if finish_reason == "tool_calls" && (tool_calls.is_none() || has_unfinished_candidates) {
                finish_reason = "malformed_tool_call".to_string();
            }
            let wrapped_arguments = reply
                .message
                .tool_calls()
                .iter()
                .filter(|call| call.arguments.json.contains(UNPARSED_ARGUMENTS_KEY))
                .count();
            let mut notes = Vec::new();
            if has_unfinished_candidates {
                notes.push("model generated unfinished tool call(s)".to_string());
            }
            if wrapped_arguments > 0 {
                notes.push(format!("{wrapped_arguments} tool call(s) with unparseable arguments"));
            }
            log.finish(&finish_reason, Some(&reply.stats), notes);
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
                        reasoning_content: reply.message.reasoning().filter(|reasoning| !reasoning.is_empty()),
                        tool_calls,
                        tool_call_id: None,
                    },
                    finish_reason,
                }],
                usage: usage_from_stats(&reply.stats, prefix_cache),
            }
        },
        None => {
            log.fail("no response generated");
            error_response(id, model, created, "No response generated")
        },
    };
    send_response(&sender, &response);
}

async fn run_stream(
    session: Arc<Mutex<ChatSession>>,
    messages: Vec<ChatMessage>,
    config: ChatReplyConfig,
    id: String,
    model: String,
    created: i64,
    prefix_cache: bool,
    sender: mpsc::UnboundedSender<Event>,
    log: RequestLog,
) {
    // Same as the blocking path: don't start generating for a client that
    // disconnected while queued on the session.
    let session = tokio::select! {
        session = session.lock() => session,
        () = sender.closed() => {
            log.fail("client disconnected");
            return;
        },
    };
    let has_tools = messages.iter().any(|message| !message.tool_namespaces().is_empty());
    let input = match prepare_input(&session, messages, prefix_cache).await {
        Ok(input) => input,
        Err(error) => {
            log.fail(&error.to_string());
            let _ = sender.send(Event::data(chunk_json(
                &id,
                &model,
                created,
                StreamDelta {
                    content: Some(format!("Error: {error}")),
                    ..StreamDelta::default()
                },
                Some("stop".to_string()),
                None,
            )));
            let _ = sender.send(Event::data("[DONE]"));
            return;
        },
    };

    let _ = sender.send(Event::data(chunk_json(
        &id,
        &model,
        created,
        StreamDelta {
            role: Some("assistant".to_string()),
            ..StreamDelta::default()
        },
        None,
        None,
    )));

    let stream = session.reply_with_stream(input, config).await;
    let mut emitted = 0usize;
    let mut emitted_reasoning = 0usize;
    let mut emitted_tool_calls = 0usize;
    let mut tool_call_streamers: HashMap<usize, ToolCallStreamer> = HashMap::new();
    let mut final_text = String::new();
    let mut finish_reason = "stop".to_string();
    let mut final_stats: Option<ChatReplyStats> = None;
    let mut final_had_candidates = false;
    let mut wrapped_arguments = 0usize;
    let mut errored = false;

    loop {
        // Rocket drops the SSE receiver as soon as the connection dies, so racing each chunk
        // against the channel closing notices a disconnect even while generation is still in
        // prefill and no chunk is on its way. Cancelling stops the backend at the next token
        // boundary and lets the session run its cancelled-turn history cleanup.
        let chunk = tokio::select! {
            chunk = stream.next() => chunk,
            () = sender.closed() => {
                cancel_and_drain(&stream).await;
                log.fail("client disconnected");
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
                let reasoning = reply.message.reasoning().unwrap_or_default();
                let reasoning_start = (emitted_reasoning..=reasoning.len())
                    .find(|&index| reasoning.is_char_boundary(index))
                    .unwrap_or(reasoning.len());
                if reasoning.len() > reasoning_start {
                    let delta = reasoning[reasoning_start..].to_string();
                    emitted_reasoning = reasoning.len();
                    let sent = sender.send(Event::data(chunk_json(
                        &id,
                        &model,
                        created,
                        StreamDelta {
                            reasoning_content: Some(delta),
                            ..StreamDelta::default()
                        },
                        None,
                        None,
                    )));
                    if sent.is_err() {
                        cancel_and_drain(&stream).await;
                        log.fail("client disconnected");
                        return;
                    }
                }
                let text = reply.message.text().unwrap_or_default();
                let start = (emitted..=text.len()).find(|&index| text.is_char_boundary(index)).unwrap_or(text.len());
                if !withhold_stream_text(has_tools, &text) && text.len() > start {
                    let delta = text[start..].to_string();
                    emitted = text.len();
                    let sent = sender.send(Event::data(chunk_json(
                        &id,
                        &model,
                        created,
                        StreamDelta {
                            content: Some(delta),
                            ..StreamDelta::default()
                        },
                        None,
                        None,
                    )));
                    if sent.is_err() {
                        cancel_and_drain(&stream).await;
                        log.fail("client disconnected");
                        return;
                    }
                }

                let finished_count = reply.message.tool_calls().len();
                let candidates = reply.message.content.iter().filter_map(|block| match block {
                    ChatContentBlock::ToolCallCandidate {
                        value,
                    } => Some(value),
                    _ => None,
                });
                for (offset, value) in candidates.enumerate() {
                    let index = finished_count + offset;
                    // candidates carry their partial text as a JSON string document, except
                    // object-valued ones (e.g. muse-glimmer's name-only progress objects)
                    let raw = serde_json::from_str::<String>(&value.json).unwrap_or_else(|_| value.json.clone());
                    let deltas =
                        tool_call_streamers.entry(index).or_insert_with(ToolCallStreamer::new).update(index, &raw);
                    for delta in deltas {
                        let sent = sender.send(Event::data(chunk_json(
                            &id,
                            &model,
                            created,
                            StreamDelta {
                                tool_calls: Some(vec![delta]),
                                ..StreamDelta::default()
                            },
                            None,
                            None,
                        )));
                        if sent.is_err() {
                            cancel_and_drain(&stream).await;
                            log.fail("client disconnected");
                            return;
                        }
                    }
                }

                let tool_calls = reply.message.tool_calls();
                while emitted_tool_calls < tool_calls.len() {
                    let index = emitted_tool_calls;
                    emitted_tool_calls += 1;
                    let call = &tool_calls[index];
                    if call.arguments.json.contains(UNPARSED_ARGUMENTS_KEY) {
                        wrapped_arguments += 1;
                    }
                    let delta = match tool_call_streamers.remove(&index) {
                        Some(mut streamer) => streamer.finish(index, call),
                        None => oai_tool_call(Some(index), call),
                    };
                    let sent = sender.send(Event::data(chunk_json(
                        &id,
                        &model,
                        created,
                        StreamDelta {
                            tool_calls: Some(vec![delta]),
                            ..StreamDelta::default()
                        },
                        None,
                        None,
                    )));
                    if sent.is_err() {
                        cancel_and_drain(&stream).await;
                        log.fail("client disconnected");
                        return;
                    }
                }

                if let Some(reason) = &reply.finish_reason {
                    finish_reason = map_finish_reason(reason);
                }
                final_stats = Some(reply.stats.clone());
                final_text = text;
                final_had_candidates = reply
                    .message
                    .content
                    .iter()
                    .any(|block| matches!(block, ChatContentBlock::ToolCallCandidate { .. }));
            },
            ChatSessionStreamChunk::Error {
                error,
            } => {
                errored = true;
                log.fail(&error.to_string());
                let _ = sender.send(Event::data(chunk_json(
                    &id,
                    &model,
                    created,
                    StreamDelta {
                        content: Some(format!("Error: {error}")),
                        ..StreamDelta::default()
                    },
                    Some("stop".to_string()),
                    None,
                )));
                break;
            },
        }
    }

    if !errored {
        // Flush withheld text that survived in the final message.
        // Text that was reclassified into tool calls is gone from it and stays suppressed.
        let start =
            (emitted..=final_text.len()).find(|&index| final_text.is_char_boundary(index)).unwrap_or(final_text.len());
        if final_text.len() > start {
            let sent = sender.send(Event::data(chunk_json(
                &id,
                &model,
                created,
                StreamDelta {
                    content: Some(final_text[start..].to_string()),
                    ..StreamDelta::default()
                },
                None,
                None,
            )));
            if sent.is_err() {
                cancel_and_drain(&stream).await;
                log.fail("client disconnected");
                return;
            }
        }
        // Same guard as the blocking path: calls the parser could not finalize must not
        // execute as a partial batch, so the turn reports a malformed tool call instead of
        // stalling on an absent or incomplete tool_calls array.
        if finish_reason == "tool_calls" && (emitted_tool_calls == 0 || final_had_candidates) {
            finish_reason = "malformed_tool_call".to_string();
        }
        let mut notes = Vec::new();
        if final_had_candidates {
            notes.push("model generated unfinished tool call(s)".to_string());
        }
        if wrapped_arguments > 0 {
            notes.push(format!("{wrapped_arguments} tool call(s) with unparseable arguments"));
        }
        log.finish(&finish_reason, final_stats.as_ref(), notes);
        let usage = final_stats
            .as_ref()
            .map_or_else(ChatCompletionUsage::default, |stats| usage_from_stats(stats, prefix_cache));
        let _ = sender.send(Event::data(chunk_json(
            &id,
            &model,
            created,
            StreamDelta::default(),
            Some(finish_reason),
            Some(usage),
        )));
    }
    let _ = sender.send(Event::data("[DONE]"));
}

#[allow(private_interfaces)]
#[post("/chat/completions", format = "json", data = "<body>")]
pub async fn handle_chat_completions(
    body: Data<'_>,
    state: &State<ServerState>,
) -> ChatCompletionResult {
    let body = match body.open(ByteUnit::Mebibyte(64)).into_string().await {
        Ok(body) if body.is_complete() => body.into_inner(),
        Ok(_) => {
            return invalid_request_response("body", "request_too_large", "request body exceeds 64 MiB".to_string());
        },
        Err(error) => {
            return invalid_request_response("body", "invalid_body", format!("failed to read request body: {error}"));
        },
    };
    let request = match serde_json::from_str::<ChatCompletionRequest>(&body) {
        Ok(request) => request,
        Err(error) => {
            RequestLog::rejected(&format!("failed to parse chat completion request: {error}"));
            return invalid_request_response(
                "body",
                "invalid_request",
                format!("failed to parse chat completion request: {error}"),
            );
        },
    };
    let id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    let created = now_unix();
    let model = state.model_name.clone();
    let is_stream = request.stream.unwrap_or(false);
    let log = RequestLog::start(
        &id,
        is_stream,
        request.messages.len(),
        request.tools.as_ref().map_or(0, Vec::len),
        request.reasoning_effort.as_ref().and_then(serde_json::Value::as_str).or(Some("unspecified")),
    );

    let config = match build_reply_config(&request) {
        Ok(config) => config,
        Err(error) => {
            log.fail(&error.message());
            return invalid_request_response("response_format", error.code(), error.message());
        },
    };
    let messages = match build_messages(&request, state.thinking_support) {
        Ok(messages) => messages,
        Err(error) => {
            let param = error.param();
            let code = error.code();
            let detail = error.into_detail();
            log.fail(&detail);
            return invalid_request_response(param, code, detail);
        },
    };

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
            state.prefix_cache,
            sender,
            log,
        ));
        let body: Pin<Box<dyn Stream<Item = Event> + Send>> = Box::pin(UnboundedReceiverStream::new(receiver));
        ChatCompletionResult::Stream(EventStream::from(body))
    } else {
        let session = Arc::clone(&state.session);
        let (sender, receiver) = mpsc::unbounded_channel::<Vec<u8>>();
        rocket::tokio::spawn(run_blocking(
            session,
            messages,
            config,
            id,
            model,
            created,
            state.prefix_cache,
            sender,
            log,
        ));
        let body: Pin<Box<dyn Stream<Item = Vec<u8>> + Send>> = Box::pin(UnboundedReceiverStream::new(receiver));
        ChatCompletionResult::Json(ByteStream::from(body))
    }
}

#[cfg(test)]
#[path = "../../unit/server/chat_completions_test.rs"]
mod tests;
