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
        session::chat::{
            ChatContentBlock, ChatMessage, ChatReplyConfig, ChatReplyFinishReason, ChatReplyStats, ChatRole,
        },
    },
};

use crate::server::ServerState;

static TOOL_KIND_FUNCTION: &str = "function";
static INCOMPLETE_TOOL_CALL_BATCH_ERROR: &str = "Model generated an incomplete tool-call batch";

#[derive(Serialize, Deserialize, Clone)]
pub struct OaiMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OaiToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OaiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OaiToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OaiToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
pub struct OaiTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OaiToolFunction,
}

#[derive(Debug, Deserialize)]
pub struct OaiToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub strict: Option<bool>,
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

#[derive(Deserialize)]
#[serde(untagged)]
enum ToolChoice {
    Mode(String),
    Function {
        #[serde(rename = "type")]
        kind: String,
        function: ToolChoiceFunction,
    },
}

#[derive(Deserialize)]
struct ToolChoiceFunction {
    name: String,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Serialize)]
struct StreamToolCall {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    function: StreamToolCallFunction,
}

#[derive(Serialize)]
struct StreamToolCallFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
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

fn to_chat_messages<'a>(
    messages: &[OaiMessage],
    tools: impl IntoIterator<Item = &'a OaiTool>,
) -> Vec<ChatMessage> {
    let mut tool_names = std::collections::HashMap::<String, String>::new();
    let mut chat_messages = messages
        .iter()
        .map(|message| {
            let role = ChatRole::from_str(&message.role).unwrap_or(ChatRole::User {});
            let mut chat_message = ChatMessage::for_role(role.clone());

            if role == (ChatRole::Tool {}) {
                let identifier = message.tool_call_id.clone();
                let name = identifier.as_ref().and_then(|identifier| tool_names.get(identifier).cloned());
                let value = message
                    .content
                    .as_deref()
                    .and_then(|content| serde_json::from_str(content).ok())
                    .unwrap_or_else(|| serde_json::Value::String(message.content.clone().unwrap_or_default()));
                chat_message = chat_message.with_block(ChatContentBlock::ToolCallResult {
                    identifier,
                    name,
                    value: Value::from(value),
                });
            } else {
                if let Some(content) = &message.content {
                    chat_message = chat_message.with_text(content.clone());
                }
                if role == (ChatRole::Assistant {}) {
                    for tool_call in message.tool_calls.as_deref().unwrap_or_default() {
                        tool_names.insert(tool_call.id.clone(), tool_call.function.name.clone());
                        chat_message = chat_message.with_tool_call(ToolCall {
                            identifier: Some(tool_call.id.clone()),
                            name: tool_call.function.name.clone(),
                            arguments: Value {
                                json: tool_call.function.arguments.clone(),
                            },
                        });
                    }
                }
            }
            chat_message
        })
        .collect::<Vec<_>>();

    let descriptions = tools
        .into_iter()
        .filter(|tool| tool.kind == TOOL_KIND_FUNCTION)
        .map(|tool| ToolDescription::Function {
            tool_function: ToolFunction {
                name: tool.function.name.clone(),
                description: tool.function.description.clone().unwrap_or_default(),
                parameters: tool.function.parameters.clone().map(Value::from),
                return_definition: None,
            },
        })
        .collect::<Vec<_>>();
    if !descriptions.is_empty() {
        let namespaces = vec![ToolNamespace {
            name: "functions".to_string(),
            description: None,
            tools: descriptions,
        }];
        if let Some(developer_message) =
            chat_messages.iter_mut().find(|message| message.role == (ChatRole::Developer {}))
        {
            developer_message.content.push(ChatContentBlock::Tools {
                namespaces,
            });
        } else {
            let definitions = ChatMessage::developer().with_tool_namespaces(namespaces);
            let position = chat_messages.iter().position(|message| message.role == (ChatRole::System {}));
            chat_messages.insert(position.map(|position| position + 1).unwrap_or(0), definitions);
        }
    }

    chat_messages
}

#[derive(Debug, PartialEq, Eq)]
enum ResponseFormatError {
    GrammarUnsupported,
    InvalidResponseFormat(String),
    InvalidJsonSchema(String),
}

#[derive(Debug, PartialEq, Eq)]
enum ToolChoiceError {
    Invalid(String),
    Unsupported(String),
    UnknownFunction(String),
}

#[derive(Debug, PartialEq, Eq)]
struct ToolDefinitionError {
    index: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct ParallelToolCallsError;

impl ParallelToolCallsError {
    fn message(&self) -> String {
        "parallel_tool_calls: false is not supported by this server when tools are enabled".to_string()
    }

    fn code(&self) -> &'static str {
        "unsupported_parallel_tool_calls"
    }
}

impl ToolDefinitionError {
    fn message(&self) -> String {
        "strict function tools are not supported by this server".to_string()
    }

    fn code(&self) -> &'static str {
        "unsupported_strict_tool"
    }

    fn param(&self) -> String {
        format!("tools[{}].function.strict", self.index)
    }
}

impl ToolChoiceError {
    fn message(&self) -> String {
        match self {
            ToolChoiceError::Invalid(detail) => format!("tool_choice is not recognized: {detail}"),
            ToolChoiceError::Unsupported(choice) => {
                format!("tool_choice {choice:?} is not supported by this server")
            },
            ToolChoiceError::UnknownFunction(name) => {
                format!("tool_choice refers to function {name:?}, which is not present in tools")
            },
        }
    }

    fn code(&self) -> &'static str {
        match self {
            ToolChoiceError::Invalid(_) | ToolChoiceError::UnknownFunction(_) => "invalid_tool_choice",
            ToolChoiceError::Unsupported(_) => "unsupported_tool_choice",
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

fn tool_choice_error_response(error: ToolChoiceError) -> ChatCompletionResult {
    invalid_request_response("tool_choice", error.code(), error.message())
}

fn tool_definition_error_response(error: ToolDefinitionError) -> ChatCompletionResult {
    invalid_request_response(&error.param(), error.code(), error.message())
}

fn parallel_tool_calls_error_response(error: ParallelToolCallsError) -> ChatCompletionResult {
    invalid_request_response("parallel_tool_calls", error.code(), error.message())
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

fn select_tools(request: &ChatCompletionRequest) -> Result<Vec<&OaiTool>, ToolChoiceError> {
    let tools = request.tools.as_deref().unwrap_or_default();
    let choice = match &request.tool_choice {
        Some(choice) => serde_json::from_value::<ToolChoice>(choice.clone())
            .map_err(|error| ToolChoiceError::Invalid(error.to_string()))?,
        None => ToolChoice::Mode("auto".to_string()),
    };

    match choice {
        ToolChoice::Mode(mode) if mode == "auto" => Ok(tools.iter().collect()),
        ToolChoice::Mode(mode) if mode == "none" => Ok(Vec::new()),
        ToolChoice::Mode(mode) if mode == "required" => Err(ToolChoiceError::Unsupported(mode)),
        ToolChoice::Mode(mode) => Err(ToolChoiceError::Invalid(format!("unknown mode {mode:?}"))),
        ToolChoice::Function {
            kind,
            function,
        } if kind == TOOL_KIND_FUNCTION => {
            if tools.iter().any(|tool| tool.kind == TOOL_KIND_FUNCTION && tool.function.name == function.name) {
                Err(ToolChoiceError::Unsupported(format!("function {}", function.name)))
            } else {
                Err(ToolChoiceError::UnknownFunction(function.name))
            }
        },
        ToolChoice::Function {
            kind,
            ..
        } => Err(ToolChoiceError::Unsupported(kind)),
    }
}

fn validate_selected_tools(
    request: &ChatCompletionRequest,
    selected_tools: &[&OaiTool],
) -> Result<(), ToolDefinitionError> {
    let declared_tools = request.tools.as_deref().unwrap_or_default();
    for tool in selected_tools {
        if tool.function.strict == Some(true) {
            let index = declared_tools.iter().position(|declared| std::ptr::eq(declared, *tool)).unwrap_or_default();
            return Err(ToolDefinitionError {
                index,
            });
        }
    }
    Ok(())
}

fn validate_parallel_tool_calls(
    request: &ChatCompletionRequest,
    selected_tools: &[&OaiTool],
) -> Result<(), ParallelToolCallsError> {
    if request.parallel_tool_calls == Some(false) && !selected_tools.is_empty() {
        return Err(ParallelToolCallsError);
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

fn response_tool_calls(message: &ChatMessage) -> Option<Vec<OaiToolCall>> {
    let tool_calls = message
        .tool_calls()
        .into_iter()
        .map(|tool_call| OaiToolCall {
            id: tool_call.identifier.unwrap_or_default(),
            kind: TOOL_KIND_FUNCTION.to_string(),
            function: OaiToolCallFunction {
                name: tool_call.name,
                arguments: tool_call.arguments.json,
            },
        })
        .collect::<Vec<_>>();
    (!tool_calls.is_empty()).then_some(tool_calls)
}

fn has_incomplete_tool_call_batch(
    message: &ChatMessage,
    finish_reason: Option<&ChatReplyFinishReason>,
) -> bool {
    let has_candidate = message.content.iter().any(|block| matches!(block, ChatContentBlock::ToolCallCandidate { .. }));
    has_candidate || (finish_reason == Some(&ChatReplyFinishReason::ToolCalls) && message.tool_calls().is_empty())
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

fn stream_tool_call_deltas(
    tool_calls: &[ToolCall],
    emitted_count: &mut usize,
) -> Vec<StreamToolCall> {
    let deltas = tool_calls
        .iter()
        .enumerate()
        .skip(*emitted_count)
        .map(|(index, tool_call)| StreamToolCall {
            index,
            id: Some(tool_call.identifier.clone().unwrap_or_default()),
            kind: Some(TOOL_KIND_FUNCTION.to_string()),
            function: StreamToolCallFunction {
                name: Some(tool_call.name.clone()),
                arguments: Some(tool_call.arguments.json.clone()),
            },
        })
        .collect::<Vec<_>>();
    *emitted_count = tool_calls.len();
    deltas
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
            Some(reply) if has_incomplete_tool_call_batch(&reply.message, reply.finish_reason.as_ref()) => {
                error_response(id, model, created, INCOMPLETE_TOOL_CALL_BATCH_ERROR)
            },
            Some(reply) => ChatCompletionResponse {
                id,
                object: "chat.completion".to_string(),
                created,
                model,
                choices: vec![ChatCompletionChoice {
                    index: 0,
                    message: OaiMessage {
                        role: "assistant".to_string(),
                        content: reply.message.text(),
                        tool_calls: response_tool_calls(&reply.message),
                        tool_call_id: None,
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
                tool_calls: None,
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
            tool_calls: None,
        },
        None,
        None,
    )));

    let stream = session.reply_with_stream(messages, config).await;
    let mut emitted = 0usize;
    let mut emitted_tool_calls = 0usize;
    let mut finish_reason = "stop".to_string();
    let mut usage = ChatCompletionUsage::default();
    let mut errored = false;
    let mut incomplete_tool_call_batch = false;

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
                            tool_calls: None,
                        },
                        None,
                        None,
                    )));
                    if sent.is_err() {
                        return;
                    }
                }
                incomplete_tool_call_batch =
                    has_incomplete_tool_call_batch(&reply.message, reply.finish_reason.as_ref());
                if let Some(reason) = &reply.finish_reason {
                    if incomplete_tool_call_batch {
                        errored = true;
                        let _ = sender.send(Event::data(chunk_json(
                            &id,
                            &model,
                            created,
                            StreamDelta {
                                role: None,
                                content: Some(format!("Error: {INCOMPLETE_TOOL_CALL_BATCH_ERROR}")),
                                tool_calls: None,
                            },
                            Some("stop".to_string()),
                            None,
                        )));
                        break;
                    }
                    let tool_calls = reply.message.tool_calls();
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
                        content: Some(format!("Error: {error}")),
                        tool_calls: None,
                    },
                    Some("stop".to_string()),
                    None,
                )));
                break;
            },
        }
    }

    if !errored && incomplete_tool_call_batch {
        errored = true;
        let _ = sender.send(Event::data(chunk_json(
            &id,
            &model,
            created,
            StreamDelta {
                role: None,
                content: Some(format!("Error: {INCOMPLETE_TOOL_CALL_BATCH_ERROR}")),
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
    let tools = match select_tools(&request) {
        Ok(tools) => tools,
        Err(error) => return tool_choice_error_response(error),
    };
    if let Err(error) = validate_selected_tools(&request, &tools) {
        return tool_definition_error_response(error);
    }
    if let Err(error) = validate_parallel_tool_calls(&request, &tools) {
        return parallel_tool_calls_error_response(error);
    }
    let messages = to_chat_messages(&request.messages, tools);

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
