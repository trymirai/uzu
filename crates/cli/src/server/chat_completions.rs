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
        basic::{Grammar, ReasoningEffort, SamplingMethod},
        session::chat::{ChatMessage, ChatReplyConfig, ChatReplyFinishReason, ChatReplyStats, ChatRole},
    },
};

use crate::{
    common::model_capabilities::ThinkingSupport,
    server::{
        ServerState,
        chat_tool_calls::{
            OaiTool, OaiToolCall, backfill_tool_result_names, choose_tools, insert_tools_message, reply_tool_calls,
            to_tool_call, tool_call_deltas, tool_call_result_block, withhold_stream_text,
        },
    },
};

#[derive(Serialize, Deserialize, Clone)]
pub struct OaiMessage {
    pub role: String,
    #[serde(default)]
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

#[derive(Serialize, Clone, Default)]
pub struct ChatCompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
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
            let mut chat_message = ChatMessage::for_role(role);
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
            // The engine reads the effort from a reasoning_effort block carried on a system message.
            messages.insert(0, ChatMessage::system().with_reasoning_effort(effort));
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

fn usage_from_stats(stats: &ChatReplyStats) -> ChatCompletionUsage {
    let prompt_tokens = stats.tokens_count_input.unwrap_or(0);
    let completion_tokens = stats.tokens_count_output.unwrap_or(0);
    ChatCompletionUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
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
                let tool_calls = reply_tool_calls(&reply.message);
                // OpenAI sets content to null (not "") on tool call replies.
                let content = reply.message.text().or_else(|| tool_calls.is_none().then(String::new));
                let mut finish_reason =
                    reply.finish_reason.as_ref().map(map_finish_reason).unwrap_or_else(|| "stop".to_string());
                // A call the parser could not finalize yields ToolCalls with nothing to execute.
                // Clients would stall on an absent tool_calls array, so report stop.
                if tool_calls.is_none() && finish_reason == "tool_calls" {
                    finish_reason = "stop".to_string();
                }
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
                content: Some(format!("Error: {error}")),
                ..StreamDelta::default()
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
            ..StreamDelta::default()
        },
        None,
        None,
    )));

    let has_tools = messages.iter().any(|message| !message.tool_namespaces().is_empty());
    let stream = session.reply_with_stream(messages, config).await;
    let mut emitted = 0usize;
    let mut emitted_tool_calls = 0usize;
    let mut final_text = String::new();
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
                        return;
                    }
                }

                let tool_calls = reply.message.tool_calls();
                if tool_calls.len() > emitted_tool_calls {
                    let delta = tool_call_deltas(&tool_calls, emitted_tool_calls);
                    emitted_tool_calls = tool_calls.len();
                    let sent = sender.send(Event::data(chunk_json(
                        &id,
                        &model,
                        created,
                        StreamDelta {
                            tool_calls: Some(delta),
                            ..StreamDelta::default()
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
                final_text = text;
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
                return;
            }
        }
        // Same guard as the blocking path: candidates that never finalized produce a ToolCalls finish with
        // no emitted tool call deltas, which reads as stop.
        if emitted_tool_calls == 0 && finish_reason == "tool_calls" {
            finish_reason = "stop".to_string();
        }
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
        Err(error) => return invalid_request_response("response_format", error.code(), error.message()),
    };
    let messages = match build_messages(&request, state.thinking_support) {
        Ok(messages) => messages,
        Err(error) => return invalid_request_response(error.param(), error.code(), error.into_detail()),
    };

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
