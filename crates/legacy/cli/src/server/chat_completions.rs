use std::{collections::HashMap, path::PathBuf, pin::Pin, str::FromStr, sync::Arc};

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
use tracing::Instrument;
use uzu::{
    session::chat::{ChatSession, ChatSessionError, ChatSessionStream, ChatSessionStreamChunk, UNPARSED_ARGUMENTS_KEY},
    types::{
        basic::{BackendError, Grammar, ReasoningEffort, SamplingMethod, SamplingSeed},
        session::chat::{
            ChatContentBlock, ChatMessage, ChatReplyConfig, ChatReplyFinishReason, ChatReplyStats, ChatRole,
        },
    },
};

use crate::{
    common::thinking::ThinkingSupport,
    server::{
        ServerState, SwapSlot, SlotSnapshot,
        chat_tool_calls::{
            OaiTool, OaiToolCall, ToolCallStreamer, ToolParameterTypes, backfill_tool_result_names, choose_tools,
            coerce_tool_call, format_lfm3_audio_assistant_tool_calls, format_lfm3_audio_tool_preamble,
            insert_tools_message, oai_tool_call, parse_lfm3_audio_tool_calls, parse_parameter_text,
            reply_tool_calls, to_tool_call, tool_call_result_block, tool_parameter_types, withhold_stream_text,
        },
        error::{OaiError, OaiErrorResponse, strip_backend_prefixes},
        model_slot::LoadedModel,
        request_info::RequestInfo,
        request_log::RequestLog,
    },
};

/// Managed Rocket state: the server-level (per-process) configuration plus the
/// model slot. Per-model state lives in [`LoadedModel`] inside the slot; chat
/// handlers snapshot the slot and work against the ready model or fail with
/// 409 `model_not_loaded`.
pub struct ChatState {
    /// Engine shared across dynamic model loads; the admin load task uses it
    /// to resolve and build models at runtime.
    pub engine: Arc<uzu::engine::Engine>,
    /// The single model slot (idle/loading/ready/failed).
    pub slot: Arc<SwapSlot>,
    /// Serializes dynamic loads: one load/swap at a time.
    pub load_gate: Arc<Mutex<()>>,
    /// Server-level flag: reuse the previous request's state when the new
    /// prompt extends it.
    pub prefix_cache: bool,
    /// Default sampling seed for requests that don't set `seed`.
    pub seed: Option<i64>,
    /// Clamped session context length applied when a model is loaded.
    pub max_context_tokens: u32,
    /// Directory scanned by GET /v1/models for loadable models.
    pub models_dir: Option<PathBuf>,
}

#[derive(Clone)]
pub struct OaiImage {
    pub url: String,
    pub image: image::DynamicImage,
}

impl std::fmt::Debug for OaiImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OaiImage")
            .field("url", &self.url)
            .field("dimensions", &(self.image.width(), self.image.height()))
            .field("color", &self.image.color())
            .finish()
    }
}

fn extract_image_url(part: &serde_json::Value) -> Result<String, String> {
    let image_url_val = part
        .get("image_url")
        .ok_or_else(|| "message content part with type 'image_url' is missing 'image_url' field".to_string())?;

    if let Some(url) = image_url_val.as_str() {
        return Ok(url.to_string());
    }
    if let Some(url) = image_url_val.get("url").and_then(serde_json::Value::as_str) {
        return Ok(url.to_string());
    }
    Err("image_url object is missing 'url' property".to_string())
}

fn extract_video_url(part: &serde_json::Value) -> Result<String, String> {
    let value = part
        .get("video_url")
        .or_else(|| part.get("video"))
        .ok_or_else(|| "message content part with type 'video_url' is missing 'video_url' field".to_string())?;
    if let Some(url) = value.as_str() {
        return Ok(url.to_string());
    }
    if let Some(url) = value.get("url").and_then(serde_json::Value::as_str) {
        return Ok(url.to_string());
    }
    Err("video_url object is missing 'url' property".to_string())
}

fn decode_image_source_local(url: &str) -> Result<OaiImage, String> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (_header, b64_payload) = rest
            .split_once(',')
            .ok_or_else(|| "invalid data URL: missing ',' payload separator".to_string())?;

        use base64::Engine;
        let clean_b64 = b64_payload.trim();
        let bytes = base64::prelude::BASE64_STANDARD
            .decode(clean_b64)
            .or_else(|_| base64::prelude::BASE64_URL_SAFE.decode(clean_b64))
            .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(clean_b64))
            .map_err(|e| format!("failed to decode base64 image data: {e}"))?;

        let image = image::load_from_memory(&bytes)
            .map_err(|e| format!("failed to decode image format: {e}"))?;

        return Ok(OaiImage {
            url: url.to_string(),
            image,
        });
    }

    if let Some(path_str) = url.strip_prefix("file://") {
        let bytes = std::fs::read(path_str)
            .map_err(|e| format!("failed to read image file from '{path_str}': {e}"))?;
        let image = image::load_from_memory(&bytes)
            .map_err(|e| format!("failed to decode image from '{path_str}': {e}"))?;
        return Ok(OaiImage {
            url: url.to_string(),
            image,
        });
    }

    let path = std::path::Path::new(url);
    if path.exists() && path.is_file() {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("failed to read image file from '{url}': {e}"))?;
        let image = image::load_from_memory(&bytes)
            .map_err(|e| format!("failed to decode image from '{url}': {e}"))?;
        return Ok(OaiImage {
            url: url.to_string(),
            image,
        });
    }

    Err(format!("unsupported or invalid image URL / path: {url}"))
}

fn remote_media_client() -> &'static reqwest::Client {
    static CLIENT: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
        reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    });
    &CLIENT
}

async fn decode_image_source_remote(url: &str) -> Result<OaiImage, String> {
    // The default client has no timeout; a slow/hung URL would pin this
    // request (and its worker) indefinitely.
    let resp = remote_media_client()
        .get(url)
        .send()
        .await
        .map_err(|e| format!("failed to download image from '{url}': {e}"))?;
    let bytes = resp.bytes()
        .await
        .map_err(|e| format!("failed to read response bytes from '{url}': {e}"))?;
    let u = url.to_string();
    let image = tokio::task::spawn_blocking(move || {
        image::load_from_memory(&bytes)
            .map_err(|e| format!("failed to decode downloaded image from '{u}': {e}"))
    })
    .await
    .map_err(|e| format!("task join error: {e}"))??;

    Ok(OaiImage {
        url: url.to_string(),
        image,
    })
}

pub async fn resolve_message_images(messages: &mut [OaiMessage]) -> Result<(), String> {
    for msg in messages.iter_mut() {
        for url in &msg.image_urls {
            let img = decode_image_source_remote(url).await?;
            msg.images.push(img);
        }
        msg.image_urls.clear();
        for url in &msg.video_urls {
            let url = url.clone();
            let frames = tokio::task::spawn_blocking(move || {
                if url.starts_with("http://") || url.starts_with("https://") {
                    let response = reqwest::blocking::get(&url).map_err(|e| format!("video download failed: {e}"))?;
                    let bytes = response.bytes().map_err(|e| format!("video download failed: {e}"))?;
                    crate::common::video::decode_video_bytes(
                        &bytes,
                        "mp4",
                        crate::common::video::VIDEO_FPS,
                        crate::common::video::VIDEO_MAX_FRAMES,
                    )
                    .map_err(|e| format!("{e}"))
                } else {
                    crate::common::video::decode_video_source(&url).map_err(|e| format!("{e}"))
                }
            })
            .await
            .map_err(|e| format!("video task join error: {e}"))??;
            msg.videos.push(frames);
        }
        msg.video_urls.clear();
    }
    Ok(())
}

#[derive(Serialize, Clone)]
pub struct OaiAudioOutput {
    pub id: String,
    pub data: String,
    pub transcript: String,
}

#[derive(Serialize, Clone)]
pub struct OaiMessage {
    pub role: String,
    pub content: Option<String>,
    // vLLM-style reasoning channel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OaiToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<OaiAudioOutput>,
    #[serde(skip_serializing)]
    pub image_urls: Vec<String>,
    #[serde(skip)]
    pub images: Vec<OaiImage>,
    #[serde(skip_serializing)]
    pub video_urls: Vec<String>,
    #[serde(skip)]
    pub videos: Vec<Vec<image::DynamicImage>>,
    #[serde(skip)]
    pub audios: Vec<Vec<u8>>,
}

impl<'de> Deserialize<'de> for OaiMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawMessage {
            role: String,
            #[serde(default)]
            content: Option<serde_json::Value>,
            #[serde(default)]
            reasoning_content: Option<String>,
            #[serde(default)]
            tool_calls: Option<Vec<OaiToolCall>>,
            #[serde(default)]
            tool_call_id: Option<String>,
        }

        let raw = RawMessage::deserialize(deserializer)?;
        let (content, image_urls, images, video_urls, audios) = match raw.content {
            None => (None, Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            Some(serde_json::Value::Null) => (None, Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            Some(serde_json::Value::String(text)) => (Some(text), Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            Some(serde_json::Value::Array(parts)) => {
                let mut text = String::new();
                let mut image_urls = Vec::new();
                let mut images = Vec::new();
                let mut video_urls = Vec::new();
                let mut audios = Vec::new();
                for part in parts {
                    match part.get("type").and_then(serde_json::Value::as_str) {
                        Some("text") => {
                            text.push_str(part.get("text").and_then(serde_json::Value::as_str).unwrap_or_default());
                        },
                        Some("image_url") => {
                            let url = extract_image_url(&part).map_err(serde::de::Error::custom)?;
                            if url.starts_with("http://") || url.starts_with("https://") {
                                image_urls.push(url);
                            } else {
                                let decoded = decode_image_source_local(&url).map_err(serde::de::Error::custom)?;
                                images.push(decoded);
                            }
                        },
                        Some("video_url") | Some("video") => {
                            let url = extract_video_url(&part).map_err(serde::de::Error::custom)?;
                            video_urls.push(url);
                        },
                        Some("input_audio") => {
                            let input_audio = part
                                .get("input_audio")
                                .ok_or_else(|| serde::de::Error::custom("content part with type 'input_audio' is missing 'input_audio' field"))?;
                            let data_str = input_audio
                                .get("data")
                                .and_then(serde_json::Value::as_str)
                                .ok_or_else(|| serde::de::Error::custom("input_audio is missing 'data' string"))?;
                            let raw_b64 = data_str.split_once(',').map_or(data_str, |(_, rest)| rest);
                            let wav = base64::Engine::decode(
                                &base64::engine::general_purpose::STANDARD,
                                raw_b64.trim(),
                            )
                            .map_err(|e| serde::de::Error::custom(format!("invalid input_audio base64: {e}")))?;
                            audios.push(wav);
                        },
                        Some(other) => {
                            return Err(serde::de::Error::custom(format!(
                                "unsupported message content part type: {other}"
                            )));
                        },
                        None => return Err(serde::de::Error::custom("message content part is missing a type")),
                    }
                }
                (Some(text), image_urls, images, video_urls, audios)
            },
            Some(_) => return Err(serde::de::Error::custom("message content must be a string or an array of content parts")),
        };

        Ok(OaiMessage {
            role: raw.role,
            content,
            reasoning_content: raw.reasoning_content,
            tool_calls: raw.tool_calls,
            tool_call_id: raw.tool_call_id,
            audio: None,
            image_urls,
            images,
            video_urls,
            videos: Vec::new(),
            audios,
        })
    }
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<OaiMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub modalities: Option<Vec<String>>,
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
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub audio_temperature: Option<f32>,
    #[serde(default)]
    pub audio_top_k: Option<usize>,
    #[serde(default)]
    pub audio_repetition_penalty: Option<f32>,
    #[serde(default)]
    pub audio_repetition_window: Option<usize>,
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
    #[serde(default)]
    pub session_id: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    audio: Option<OaiAudioOutput>,
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
    session_id: Option<String>,
}

pub enum ChatCompletionResult {
    // The single JSON body arrives as a stream: Rocket only surfaces a client
    // disconnect by dropping the body stream, which is the blocking path's one
    // signal to cancel a generation nobody is waiting for.
    Json(Status, ByteStream<Pin<Box<dyn Stream<Item = Vec<u8>> + Send>>>, Option<String>),
    Stream(EventStream<Pin<Box<dyn Stream<Item = Event> + Send>>>, Option<String>),
    Error(status::Custom<Json<OaiErrorResponse>>),
    /// 409 when no (or a different) model is resident in the slot.
    ModelNotLoaded(status::Custom<Json<ModelNotLoadedBody>>),
}

impl ChatCompletionResult {
    pub fn json(status: Status, body: ByteStream<Pin<Box<dyn Stream<Item = Vec<u8>> + Send>>>) -> Self {
        ChatCompletionResult::Json(status, body, None)
    }

    pub fn json_with_session(
        status: Status,
        body: ByteStream<Pin<Box<dyn Stream<Item = Vec<u8>> + Send>>>,
        session_id: String,
    ) -> Self {
        ChatCompletionResult::Json(status, body, Some(session_id))
    }

    pub fn stream(stream: EventStream<Pin<Box<dyn Stream<Item = Event> + Send>>>) -> Self {
        ChatCompletionResult::Stream(stream, None)
    }

    pub fn stream_with_session(
        stream: EventStream<Pin<Box<dyn Stream<Item = Event> + Send>>>,
        session_id: String,
    ) -> Self {
        ChatCompletionResult::Stream(stream, Some(session_id))
    }
}

impl<'r> Responder<'r, 'r> for ChatCompletionResult {
    fn respond_to(
        self,
        request: &'r Request<'_>,
    ) -> rocket::response::Result<'r> {
        match self {
            ChatCompletionResult::Json(status, body, session_id) => {
                let mut response = (ContentType::JSON, body).respond_to(request)?;
                response.set_status(status);
                if let Some(sid) = session_id {
                    response.set_raw_header("X-Session-ID", sid);
                }
                Ok(response)
            },
            ChatCompletionResult::Stream(stream, session_id) => {
                let mut response = stream.respond_to(request)?;
                if let Some(sid) = session_id {
                    response.set_raw_header("X-Session-ID", sid);
                }
                Ok(response)
            },
            ChatCompletionResult::Error(error) => error.respond_to(request),
            ChatCompletionResult::ModelNotLoaded(error) => error.respond_to(request),
        }
    }
}

pub(crate) fn to_chat_messages(messages: &[OaiMessage]) -> Vec<ChatMessage> {
    merge_adjacent_system_messages(messages.iter().map(to_chat_message))
}

fn to_chat_message(message: &OaiMessage) -> ChatMessage {
    let mut chat_message = ChatMessage::for_role(to_chat_role(&message.role));
    // Block order matches how the session stores generated replies
    // (reasoning, then text, then tool calls) so a client that echoes
    // reasoning back keeps the message prefix intact.
    if let Some(reasoning) = message.reasoning_content.as_ref().filter(|reasoning| !reasoning.is_empty()) {
        chat_message = chat_message.with_reasoning(reasoning.clone());
    }
    for img in &message.images {
        chat_message = chat_message.with_block(ChatContentBlock::Image {
            url: img.url.clone(),
        });
    }
    if let Some(identifier) = &message.tool_call_id {
        let result = tool_call_result_block(identifier, message.content.clone().unwrap_or_default());
        chat_message = chat_message.with_block(result);
    } else if let Some(content) = message.content.as_ref().filter(|content| !content.is_empty()) {
        chat_message = chat_message.with_text(content.clone());
    }
    for tool_call in message.tool_calls.iter().flatten() {
        chat_message = chat_message.with_tool_call(to_tool_call(tool_call));
    }
    chat_message
}

// OpenAI clients send their instructions as `developer` (the system role of o1 and
// newer; OpenAI treats it as system for other models). The engine's developer slot
// carries only the tool declarations (see insert_tools_message), so client
// instructions go to the system message.
fn to_chat_role(role: &str) -> ChatRole {
    match ChatRole::from_str(role).unwrap_or(ChatRole::User {}) {
        ChatRole::Developer {} => ChatRole::System {},
        role => role,
    }
}

// A client may send both system and developer instructions, but templates reject two
// system messages in a row. Their texts join into one block: with two, templates get a
// list instead of a string and Harmony concatenates them without a separator.
fn merge_adjacent_system_messages(messages: impl Iterator<Item = ChatMessage>) -> Vec<ChatMessage> {
    let mut merged: Vec<ChatMessage> = Vec::new();
    for message in messages {
        match merged.last_mut() {
            Some(previous) if previous.role == ChatRole::System {} && message.role == ChatRole::System {} => {
                append_system_content(previous, message.content);
            },
            _ => merged.push(message),
        }
    }
    merged
}

fn append_system_content(
    message: &mut ChatMessage,
    content: Vec<ChatContentBlock>,
) {
    for block in content {
        let existing_text = message.content.iter_mut().find_map(|block| match block {
            ChatContentBlock::Text {
                value,
            } => Some(value),
            _ => None,
        });
        match (block, existing_text) {
            (
                ChatContentBlock::Text {
                    value,
                },
                Some(existing_text),
            ) => {
                existing_text.push_str("\n\n");
                existing_text.push_str(&value);
            },
            (block, _) => message.content.push(block),
        }
    }
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
    InvalidResponseFormat(String),
    InvalidJsonSchema(String),
}

impl ResponseFormatError {
    fn message(&self) -> String {
        match self {
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

/// 409 body for slot mismatches: no model resident, a different model
/// resident, or the slot is loading/failed. `hint` tells API clients how to
/// recover (dynamic-load admin API).
#[derive(Serialize)]
pub struct ModelNotLoadedBody {
    pub error: ModelNotLoadedError,
}

#[derive(Serialize)]
pub struct ModelNotLoadedError {
    #[serde(rename = "type")]
    pub kind: String,
    pub requested: Option<String>,
    pub current: Option<String>,
    pub hint: String,
}

pub(crate) fn model_not_loaded_response(
    requested: Option<&str>,
    current: Option<&str>,
    detail: &str,
) -> ChatCompletionResult {
    ChatCompletionResult::ModelNotLoaded(status::Custom(
        Status::Conflict,
        Json(model_not_loaded_body(requested, current, detail)),
    ))
}

/// The 409 body alone, for routes with their own response enum (raw completions).
pub(crate) fn model_not_loaded_body(
    requested: Option<&str>,
    current: Option<&str>,
    detail: &str,
) -> ModelNotLoadedBody {
    ModelNotLoadedBody {
        error: ModelNotLoadedError {
            kind: "model_not_loaded".to_string(),
            requested: requested.map(str::to_owned),
            current: current.map(str::to_owned),
            hint: format!(
                "POST /v1/models/load with {{\"model\": \"<name>\"}}, then poll GET /v1/models/status ({detail})"
            ),
        },
    }
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

/// Wrong-typed values already fail JSON extraction; out-of-range numbers would
/// otherwise reach the sampler unchecked (a huge temperature generates garbage
/// with HTTP 200).
fn validate_sampling(request: &ChatCompletionRequest) -> Result<(), (&'static str, String)> {
    if let Some(temperature) = request.temperature
        && !(0.0..=2.0).contains(&temperature)
    {
        return Err(("temperature", format!("temperature must be between 0 and 2, got {temperature}")));
    }
    if let Some(top_p) = request.top_p
        && !(top_p > 0.0 && top_p <= 1.0)
    {
        return Err(("top_p", format!("top_p must be greater than 0 and at most 1, got {top_p}")));
    }
    if let Some(top_k) = request.top_k
        && top_k < 1
    {
        return Err(("top_k", format!("top_k must be at least 1, got {top_k}")));
    }
    Ok(())
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
    if let Some(seed) = request.seed {
        config = config.with_sampling_seed(SamplingSeed::Custom {
            seed,
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
        Some(ResponseFormat::JsonObject) => config.with_grammar(Some(Grammar::JsonAny {})),
        Some(ResponseFormat::JsonSchema {
            json_schema,
        }) => config.with_grammar(Some(json_schema_grammar(&json_schema)?)),
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

pub(crate) fn oai_error_body(
    status: Status,
    code: &str,
    message: &str,
) -> OaiErrorResponse {
    let kind = if status.code < 500 {
        "invalid_request_error"
    } else {
        "server_error"
    };
    OaiErrorResponse {
        error: OaiError {
            message: strip_backend_prefixes(message).to_string(),
            kind: kind.to_string(),
            param: None,
            code: Some(code.to_string()),
        },
    }
}

/// OpenAI clients compact the conversation on context_length_exceeded, so an
/// overflow must not reach them as a generic backend_error.
fn stream_error_body(error: &ChatSessionError) -> OaiErrorResponse {
    match error {
        ChatSessionError::Backend {
            error: BackendError::ContextOverflow {},
        } => {
            let mut body = oai_error_body(
                Status::BadRequest,
                "context_length_exceeded",
                "Input exceeds the context window of this model",
            );
            body.error.param = Some("messages".to_string());
            body
        },
        _ => oai_error_body(Status::BadRequest, "backend_error", &error.to_string()),
    }
}

fn send_error_body(
    sender: &mpsc::UnboundedSender<Vec<u8>>,
    body: &OaiErrorResponse,
) {
    let _ = sender.send(serde_json::to_vec(body).unwrap_or_default());
}

fn chunk_json(
    id: &str,
    model: &str,
    created: i64,
    delta: StreamDelta,
    finish_reason: Option<String>,
    usage: Option<ChatCompletionUsage>,
) -> String {
    chunk_json_with_session(id, model, created, delta, finish_reason, usage, None)
}

fn chunk_json_with_session(
    id: &str,
    model: &str,
    created: i64,
    delta: StreamDelta,
    finish_reason: Option<String>,
    usage: Option<ChatCompletionUsage>,
    session_id: Option<&str>,
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
        session_id: session_id.map(|s| s.to_string()),
    };
    serde_json::to_string(&chunk).unwrap_or_default()
}

/// Schema coercion retypes arguments on the wire, so a client may echo `5` or
/// `[1,2]` where the session stored the text `"5"` or `"[1, 2]"` (or a JSON
/// string where the session stored the value it spells); such pairs must still
/// count as the same prefix.
fn tool_call_arguments_equivalent(
    stored: &str,
    echoed: &str,
) -> bool {
    if stored == echoed {
        return true;
    }
    match (serde_json::from_str::<serde_json::Value>(stored), serde_json::from_str::<serde_json::Value>(echoed)) {
        (Ok(stored), Ok(echoed)) => json_values_equivalent(&stored, &echoed),
        _ => false,
    }
}

fn json_values_equivalent(
    left: &serde_json::Value,
    right: &serde_json::Value,
) -> bool {
    use serde_json::Value;
    match (left, right) {
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .all(|(key, value)| right.get(key).is_some_and(|other| json_values_equivalent(value, other)))
        },
        (Value::String(text), other) | (other, Value::String(text)) if !other.is_string() => {
            parse_parameter_text(text).is_some_and(|parsed| parsed == *other)
        },
        (left, right) => left == right,
    }
}

/// Clients may omit assistant reasoning when echoing a previous turn, keep the session's reasoning in that case.
/// Tool call identifiers are assigned per turn and never rendered into tokens, so they are excluded from the comparison.
/// All other content must match.
fn messages_have_prefix(
    messages: &[ChatMessage],
    prefix: &[ChatMessage],
) -> bool {
    prefix.len() <= messages.len()
        && prefix.iter().zip(messages).all(|(prefix_message, message)| {
            let omitted_reasoning = matches!(message.role, ChatRole::Assistant {})
                && !message.content.iter().any(|block| matches!(block, ChatContentBlock::Reasoning { .. }));
            let prefix_content = prefix_message
                .content
                .iter()
                .filter(|block| !omitted_reasoning || !matches!(block, ChatContentBlock::Reasoning { .. }));
            prefix_message.role == message.role
                && prefix_message.metadata == message.metadata
                && prefix_content.clone().count() == message.content.len()
                && prefix_content.zip(&message.content).all(|(prefix_block, block)| match (prefix_block, block) {
                    (
                        ChatContentBlock::ToolCall {
                            value: prefix_call,
                        },
                        ChatContentBlock::ToolCall {
                            value: call,
                        },
                    ) => {
                        prefix_call.name == call.name
                            && tool_call_arguments_equivalent(&prefix_call.arguments.json, &call.arguments.json)
                    },
                    _ => prefix_block == block,
                })
        })
}

/// Returns only the messages the session has not seen yet when the request
/// extends the session's current history; otherwise resets the session and
/// returns the full list. Retaining the session history also retains assistant
/// reasoning omitted by the client. Token-level reuse itself is decided inside
/// the session, which falls back to a full prefill for incompatible boundaries
/// (including truncated replies) or divergent rendered tokens.
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
    parameter_types: ToolParameterTypes,
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
            send_error_body(&sender, &oai_error_body(Status::InternalServerError, "backend_error", &error.to_string()));
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
            } => {
                final_replies = replies;
            },
            ChatSessionStreamChunk::ToolResults {
                ..
            } => {},
            ChatSessionStreamChunk::Error {
                error,
            } => {
                log.fail(&error.to_string());
                send_error_body(&sender, &stream_error_body(&error));
                return;
            },
        }
    }

    let Some(reply) = final_replies.last() else {
        log.fail("no response generated");
        send_error_body(&sender, &oai_error_body(Status::InternalServerError, "no_response", "No response generated"));
        return;
    };
    let tool_calls = reply_tool_calls(&reply.message, &parameter_types);
    // OpenAI sets content to null (not "") on tool call replies.
    let content = reply.message.text().or_else(|| tool_calls.is_none().then(String::new));
    let mut finish_reason = reply.finish_reason.as_ref().map(map_finish_reason).unwrap_or_else(|| "stop".to_string());
    // Calls the parser could not finalize must not execute as a partial batch.
    // OpenAI has no finish reason for this; providers signal terminal errors with
    // unrecognized reasons (pi maps them to a thrown provider error), so the turn
    // fails loudly instead of stalling on an absent or incomplete tool_calls array.
    let has_unfinished_candidates =
        reply.message.content.iter().any(|block| matches!(block, ChatContentBlock::ToolCallCandidate { .. }));
    if finish_reason == "tool_calls" && (tool_calls.is_none() || has_unfinished_candidates) {
        finish_reason = "malformed_tool_call".to_string();
    }
    let wrapped_arguments =
        reply.message.tool_calls().iter().filter(|call| call.arguments.json.contains(UNPARSED_ARGUMENTS_KEY)).count();
    let mut notes = Vec::new();
    if has_unfinished_candidates {
        notes.push("model generated unfinished tool call(s)".to_string());
    }
    if wrapped_arguments > 0 {
        notes.push(format!("{wrapped_arguments} tool call(s) with unparseable arguments"));
    }
    log.finish(&finish_reason, Some(&reply.stats), notes);
    let response = ChatCompletionResponse {
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
                audio: None,
                image_urls: Vec::new(),
                images: Vec::new(),
                video_urls: Vec::new(),
                videos: Vec::new(),
                audios: Vec::new(),
            },
            finish_reason,
        }],
        usage: usage_from_stats(&reply.stats, prefix_cache),
        session_id: None,
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
    parameter_types: ToolParameterTypes,
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
            // The SSE response is already committed as 200, so the error travels
            // as an OpenAI-style error event, the way OpenAI reports mid-stream.
            let body = oai_error_body(Status::InternalServerError, "backend_error", &error.to_string());
            let _ = sender.send(Event::data(serde_json::to_string(&body).unwrap_or_default()));
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
                        ..
                    } => Some(value),
                    _ => None,
                });
                for (offset, value) in candidates.enumerate() {
                    let index = finished_count + offset;
                    // Candidates carry partial markup as a JSON string document or a JSON object.
                    let raw = serde_json::from_str::<String>(&value.json).unwrap_or_else(|_| value.json.clone());
                    let deltas = tool_call_streamers.entry(index).or_insert_with(ToolCallStreamer::new).update(
                        index,
                        &raw,
                        &parameter_types,
                    );
                    let deltas = match deltas {
                        Ok(deltas) => deltas,
                        Err(error) => {
                            let message = error.to_string();
                            let _ = sender.send(Event::data(
                                serde_json::to_string(&oai_error_body(Status::BadRequest, "backend_error", &message))
                                    .unwrap_or_default(),
                            ));
                            cancel_and_drain(&stream).await;
                            log.fail(&message);
                            return;
                        },
                    };
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
                    if tool_calls[index].arguments.json.contains(UNPARSED_ARGUMENTS_KEY) {
                        wrapped_arguments += 1;
                    }
                    let call = coerce_tool_call(&tool_calls[index], &parameter_types);
                    let delta = match tool_call_streamers.remove(&index) {
                        Some(mut streamer) => match streamer.finish(index, &call) {
                            Ok(delta) => delta,
                            Err(error) => {
                                let message = error.to_string();
                                let _ = sender.send(Event::data(
                                    serde_json::to_string(&oai_error_body(
                                        Status::BadRequest,
                                        "backend_error",
                                        &message,
                                    ))
                                    .unwrap_or_default(),
                                ));
                                cancel_and_drain(&stream).await;
                                log.fail(&message);
                                return;
                            },
                        },
                        None => oai_tool_call(Some(index), &call),
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
            ChatSessionStreamChunk::ToolResults {
                ..
            } => {},
            ChatSessionStreamChunk::Error {
                error,
            } => {
                errored = true;
                log.fail(&error.to_string());
                // Committed 200 stream: report as an OpenAI-style error event.
                let body = stream_error_body(&error);
                let _ = sender.send(Event::data(serde_json::to_string(&body).unwrap_or_default()));
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

/// Builds the tool-declaration preamble exactly as the chat template renders it: the
/// declarations as single-line JSON objects between <tools> tags, followed by the
/// markup format contract the model must follow when calling them.
fn tools_preamble_text(vlm: &uzu_engine::vision::AnyVlmPipeline, tools: &[&OaiTool]) -> String {
    let declarations = tools
        .iter()
        .map(|tool| {
            let mut function = serde_json::json!({
                "name": tool.function.name,
                "description": tool.function.description,
            });
            if let Some(parameters) = &tool.function.parameters {
                function["parameters"] =
                    serde_json::from_value(parameters.json.parse().unwrap_or(serde_json::Value::Null))
                        .unwrap_or(serde_json::Value::Null);
            }
            serde_json::json!({
                "type": "function",
                "function": function,
            })
            .to_string()
        })
        .collect::<Vec<_>>();
    vlm.tools_preamble(&declarations)
}

/// Splits a plain-path generation into (reasoning, content), mirroring the template's
/// think-block placement. History summaries store the content side so echo clients
/// that move reasoning into reasoning_content still prefix-cache match.
fn split_think_text(full_text: &str) -> (Option<String>, String) {
    if full_text.contains("</think>") {
        let parts: Vec<&str> = full_text.split("</think>").collect();
        let reasoning = parts[0].replace("<think>", "").trim().to_string();
        let body = parts[1..].join("</think>").trim_start_matches('\n').to_string();
        (
            if !reasoning.is_empty() {
                Some(reasoning)
            } else {
                None
            },
            body,
        )
    } else if full_text.starts_with("<think>") {
        let reasoning = full_text.replace("<think>", "").trim().to_string();
        (Some(reasoning), String::new())
    } else {
        (None, full_text.to_string())
    }
}

/// Template-contract repair: the model ended the turn while the prefilled
/// `<think>` section was still open (the generated text never contains
/// `</think>`), leaving everything it wrote inside the think block.
/// Observed on MiniCPM short-answer prompts: `<|im_start|>assistant\n<think>\n`
/// → "the answer" → EOS with no `</think>`. When the turn ended cleanly
/// (`stop`, not a truncation), that content is the reply, not reasoning —
/// a truncated turn would leave the section open too, but with a
/// `length`/`cancelled` finish reason, which we do not promote.
/// Returns `(reasoning_content, content, promoted)`.
fn promote_unclosed_think(
    reasoning: String,
    text: String,
    finish_reason: &str,
    generated_text: &str,
) -> (Option<String>, String, bool) {
    let should_promote = finish_reason == "stop"
        && text.is_empty()
        && !reasoning.is_empty()
        && !generated_text.contains("</think>");
    if should_promote {
        tracing::warn!(
            "model ended turn inside an open <think> section; promoting think content to the reply body"
        );
        (None, reasoning, true)
    } else {
        (Some(reasoning).filter(|reasoning| !reasoning.is_empty()), text, false)
    }
}

/// Splits one streamed chunk into (reasoning, content). `in_thinking` starts
/// true when the prompt opened a live think block (enable_thinking), in which
/// case the opening marker is in the prompt and only `</think>` appears in the
/// stream.
fn split_think_chunk(
    chunk: String,
    in_thinking: &mut bool,
) -> (Option<String>, Option<String>) {
    if !*in_thinking {
        if chunk.contains("<think>") {
            *in_thinking = true;
            let text = chunk.replace("<think>", "");
            return (Some(text).filter(|text| !text.is_empty()), None);
        }
        return (None, Some(chunk));
    }
    if let Some((reasoning, content)) = chunk.split_once("</think>") {
        *in_thinking = false;
        let content = content.trim_start_matches('\n');
        return (
            Some(reasoning.to_string()).filter(|text| !text.is_empty()),
            (!content.is_empty()).then(|| content.to_string()),
        );
    }
    (Some(chunk).filter(|text| !text.is_empty()), None)
}

async fn handle_vlm_completion(
    request: &ChatCompletionRequest,
    loaded: &Arc<LoadedModel>,
    model: &str,
    created: i64,
    id: &str,
    is_stream: bool,
    enable_thinking: bool,
    log: RequestLog,
    request_info: &RequestInfo,
) -> ChatCompletionResult {
    let vlm = loaded.vlm_pipeline.as_ref().unwrap();
    let session_guard = loaded.session.lock().await;
    let Some(backend_inst) = session_guard.chat_token_backend_instance().await else {
        return invalid_request_response("model", "vlm_error", "No backend instance available".to_string());
    };
    let Some(uzu_backend) = backend_inst.as_any().downcast_ref::<uzu::engine::bridge::UzuChatTokenBackendInstance<uzu_engine::backends::metal::Metal>>() else {
        return invalid_request_response("model", "vlm_error", "Backend is not Metal".to_string());
    };
    let lm = uzu_backend.language_model().clone();
    drop(session_guard);

    if let Err((param, msg)) = validate_sampling(request) {
        log.fail(&format!("Sampling validation error on {param}: {msg}"));
        return invalid_request_response(param, "invalid_request_error", msg);
    }

    let tools = match choose_tools(request.tools.as_deref(), request.tool_choice.as_ref()) {
        Ok(tools) => tools,
        Err(message) => {
            log.fail(&message);
            return invalid_request_response("tool_choice", "invalid_request_error", message);
        },
    };
    let has_tools = !tools.is_empty();
    let parameter_types = tool_parameter_types(request.tools.as_deref());
    let tools_preamble = has_tools.then(|| tools_preamble_text(vlm, &tools));

    // enable_thinking is resolved by the caller (default off when unspecified) and
    // decides whether the prefill suffix opens a live think block or a closed one.

    let stop_token_ids: Vec<u64> = lm.generation_config().stop_token_ids.to_vec();
    let stop_token_ids_i32: Vec<i32> = stop_token_ids.iter().map(|&t| t as i32).collect();
    let grammar = match &request.response_format {
        Some(val) => {
            let fmt = serde_json::from_value::<ResponseFormat>(val.clone()).ok();
            match fmt {
                Some(ResponseFormat::JsonObject) => {
                    let cfg = uzu_engine::engine::language_model::grammar::GrammarConfig::BuiltinJson;
                    uzu_engine::engine::language_model::grammar::Grammar::new(&cfg, lm.tokenizer(), None, Some(&stop_token_ids_i32)).ok()
                },
                Some(ResponseFormat::JsonSchema { json_schema }) => {
                    let schema_str = json_schema.schema.to_string();
                    let cfg = uzu_engine::engine::language_model::grammar::GrammarConfig::JsonSchema {
                        schema: schema_str,
                        any_whitespace: true,
                        indent: None,
                        separators: None,
                        strict_mode: false,
                    };
                    uzu_engine::engine::language_model::grammar::Grammar::new(&cfg, lm.tokenizer(), None, Some(&stop_token_ids_i32)).ok()
                },
                _ => None,
            }
        },
        None => None,
    };

    let sampling_config = uzu_engine::vision::VlmSamplingConfig {
        temperature: request.temperature.map(|t| t as f32),
        min_p: None,
        top_p: request.top_p.map(|p| p as f32),
        top_k: request.top_k.map(|k| k.max(0) as usize),
        repetition_penalty: request.repetition_penalty.map(|r| r as f32),
        seed: request.seed.map(|s| s as u64),
        grammar,
        vocab_limit: None,
    };
    let mut sampler = uzu_engine::vision::VlmSampler::new(sampling_config);
    // Decoder logits can be padded past the tokenizer vocabulary; never sample
    // those ids.
    sampler.set_vocab_limit(lm.tokenizer().get_vocab_size(true));

    let max_tokens = request.max_completion_tokens.or(request.max_tokens).unwrap_or(256) as usize;

    let multimodal_messages: Vec<uzu_engine::vision::MultimodalMessage> = request
        .messages
        .iter()
        .map(|m| uzu_engine::vision::MultimodalMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            reasoning_content: m.reasoning_content.clone(),
            images: m.images.iter().map(|img| img.image.clone()).collect(),
            videos: m.videos.clone(),
            tool_calls: m
                .tool_calls
                .iter()
                .flatten()
                .map(|call| uzu_engine::vision::MultimodalToolCall {
                    name: call.function.name.clone(),
                    arguments: call.function.arguments.clone(),
                })
                .collect(),
            tool_call_id: m.tool_call_id.clone(),
        })
        .collect();

    let incoming_summary = crate::server::vlm_session::VlmSessionManager::summarize_messages(&multimodal_messages);
    let requested_session_id = request
        .session_id
        .as_deref()
        .or(request_info.session_id.as_deref());

    let (session_arc, is_brand_new, active_session_id) = match loaded
        .vlm_sessions
        .get_or_create(requested_session_id, &incoming_summary, vlm, &lm, loaded.max_context_tokens)
        .await
    {
        Ok(res) => res,
        Err(err) => {
            log.fail(&err.to_string());
            return invalid_request_response("model", "vlm_error", format!("Failed to acquire VLM session: {err}"));
        }
    };

    let mut session = session_arc.clone().lock_owned().await;

    // Check if this request is a cacheable follow-up turn on top of existing session state
    let tool_result_tail = !is_brand_new
        && incoming_summary.len() > session.history.len()
        && incoming_summary[..session.history.len()] == session.history[..]
        && incoming_summary[session.history.len()..].iter().all(|m| m.role == "tool");
    let can_do_tool_followup = tool_result_tail && session.last_generation_tool_call;
    let can_do_followup = !is_brand_new
        && !session.history.is_empty()
        && session.state.context_length() > 0
        && (
            // A: Standard client repeating history: incoming starts with session.history and ends with 1 text message
            (incoming_summary.len() == session.history.len() + 1
                && incoming_summary[..session.history.len()] == session.history[..]
                && incoming_summary.last().is_some_and(|m| m.image_hashes.is_empty()))
            // B: Smart client passing session_id with only 1 text user message
            || (requested_session_id.is_some()
                && incoming_summary.len() == 1
                && incoming_summary[0].role == "user"
                && incoming_summary[0].image_hashes.is_empty())
            // C: Tool results for the call this session just generated: the tail closes the
            // assistant turn and appends the rendered tool responses as one user message.
            || can_do_tool_followup
        );

    let history_len_before_turn = session.history.len();
    let tool_tail_summaries = can_do_tool_followup.then(|| incoming_summary[history_len_before_turn..].to_vec());
    let turn_query = if can_do_tool_followup {
        let contents: Vec<&str> = incoming_summary[history_len_before_turn..]
            .iter()
            .map(|summary| summary.content.as_deref().unwrap_or_default())
            .collect();
        uzu_engine::vision::render_tool_response_text(&contents)
    } else {
        incoming_summary.last().and_then(|m| m.content.clone()).unwrap_or_default()
    };

    // The reply parser must see the exact prefill suffix so its section state (the open
    // assistant frame and any prefilled think block) matches what the model continues.
    let suffix_prompt = if enable_thinking {
        "<|im_start|>assistant\n<think>\n"
    } else {
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    };
    let mut reply_parser = if has_tools {
        match crate::server::vlm_reply::VlmReplyParser::new(loaded.reply_encoding.as_ref(), lm.tokenizer().clone()) {
            Ok(mut parser) => match parser.push_text(suffix_prompt) {
                Ok(()) => Some(parser),
                Err(error) => {
                    log.fail(&error);
                    return invalid_request_response(
                        "model",
                        "vlm_error",
                        format!("VLM reply parser init failed: {error}"),
                    );
                },
            },
            Err(error) => {
                log.fail(&error);
                return invalid_request_response(
                    "model",
                    "vlm_error",
                    format!("VLM reply parser init failed: {error}"),
                );
            },
        }
    } else {
        None
    };

    let (first_token, prompt_tokens_count, cached_tokens, followup_query) = if can_do_followup {
        let query = turn_query.clone();
        let cached = session.prompt_tokens_count;
        tracing::info!(
            "⚡ [VLM Prefix Cache Hit] Session '{}' reusing {} tokens (Vision Tower: 0.000s)",
            active_session_id,
            cached
        );
        let t2 = std::time::Instant::now();
        let tok = match tokio::task::block_in_place(|| {
            vlm.prefill_followup_turn(&lm, &mut session.state, &query, lm.tokenizer(), &mut sampler, enable_thinking)
        }) {
            Ok(t) => t,
            Err(err) => {
                log.fail(&err.to_string());
                return invalid_request_response("model", "vlm_error", format!("VLM follow-up prefill failed: {err}"));
            }
        };
        let t_prefill = t2.elapsed();
        let delta_tokens = match vlm.format_followup_tokens(&query, lm.tokenizer(), enable_thinking) {
            Ok(t) => t.len(),
            Err(_) => 10,
        };
        tracing::info!(
            "[VLM Followup Timings] session: {}, prefill: {:.3}s, cached_tokens: {}, new_tokens: {}",
            active_session_id,
            t_prefill.as_secs_f64(),
            cached,
            delta_tokens
        );
        (tok, delta_tokens, cached as u32, Some(query))
    } else {
        if !is_brand_new {
            let _ = session.state.reset();
        }
        let t0 = std::time::Instant::now();
        // Metal prefill blocks on command-buffer completion for seconds; run it under
        // block_in_place so the scheduler can move other connection tasks off this
        // worker thread instead of starving them for the whole prefill.
        let prepared = match tokio::task::block_in_place(|| {
            vlm.prepare_conversation_with_tools(
                &multimodal_messages,
                lm.tokenizer(),
                enable_thinking,
                tools_preamble.as_deref(),
            )
        }) {
            Ok(p) => p,
            Err(err) => {
                log.fail(&err.to_string());
                return invalid_request_response("image", "vlm_error", format!("VLM preprocessing failed: {err}"));
            }
        };
        let t_prep = t0.elapsed();

        let t2 = std::time::Instant::now();
        let tok = match tokio::task::block_in_place(|| vlm.prefill_with_state(&lm, &mut session.state, &prepared, &mut sampler)) {
            Ok(t) => t,
            Err(err) => {
                log.fail(&err.to_string());
                return invalid_request_response("model", "vlm_error", format!("VLM prefill failed: {err}"));
            }
        };
        let t_prefill = t2.elapsed();
        let pt_len = prepared.prompt_tokens().len();
        tracing::info!(
            "[VLM Cold Start Timings] session: {}, prepare_conversation: {:.3}s, prefill: {:.3}s (prompt_tokens: {})",
            active_session_id,
            t_prep.as_secs_f64(),
            t_prefill.as_secs_f64(),
            pt_len
        );
        (tok, pt_len, 0u32, None)
    };

    if is_stream {
        let (sender, receiver) = mpsc::unbounded_channel::<Event>();
        let vlm_pipeline = Arc::clone(vlm);
        let id_str = id.to_string();
        let model_str = model.to_string();
        let session_id_str = active_session_id.clone();

        tokio::spawn(async move {
            let mut curr_token = first_token;
            let mut tokens_count = 0;
            let tokenizer = lm.tokenizer();
            let mut decoder = uzu_engine::vision::StreamingTokenDecoder::new(tokenizer, false);
            let mut in_thinking = enable_thinking;
            let mut generated_tokens = vec![first_token];

            // Tool requests stream through the engine's parsing stack (hanashi config):
            // cumulative reasoning/text sections plus finalized/candidate tool calls.
            // Plain requests keep the direct decoder emission below.
            let mut emitted = 0usize;
            let mut emitted_reasoning = 0usize;
            let mut emitted_tool_calls = 0usize;
            let mut tool_call_streamers: HashMap<usize, ToolCallStreamer> = HashMap::new();
            let mut finish_reason = "stop".to_string();
            let mut errored = false;

            if let Some(parser) = reply_parser.as_mut() {
                // The reply frame was opened by push_text; feed the first sampled token too.
                if let Err(err) = parser.push_sampled_token(curr_token as u32) {
                    tracing::error!("VLM reply parser error: {err}");
                    errored = true;
                }
            } else if let Ok(Some(chunk)) = decoder.step(curr_token as u32) {
                let (reasoning_content, content) = split_think_chunk(chunk, &mut in_thinking);
                let delta = StreamDelta {
                    reasoning_content,
                    content,
                    ..Default::default()
                };
                let _ = sender.send(Event::data(chunk_json_with_session(
                    &id_str,
                    &model_str,
                    created,
                    delta,
                    None,
                    None,
                    Some(&session_id_str),
                )));
            }
            tokens_count += 1;

            while tokens_count < max_tokens && !stop_token_ids.contains(&curr_token) {
                match vlm_pipeline.decode_step_with_state(&lm, &mut session.state, curr_token, &mut sampler) {
                    Ok(next_tok) => {
                        curr_token = next_tok;
                        if stop_token_ids.contains(&curr_token) {
                            break;
                        }
                        generated_tokens.push(curr_token);
                        // The decode step blocks on the GPU for tens of milliseconds with no
                        // await point; yield so Rocket's connection writer on this worker can
                        // flush the SSE chunks instead of starving until the stream ends.
                        tokio::task::yield_now().await;
                        if let Some(parser) = reply_parser.as_mut() {
                            match parser.push_sampled_token(curr_token as u32) {
                                Ok(()) => {},
                                Err(err) => {
                                    tracing::error!("VLM reply parser error: {err}");
                                    errored = true;
                                },
                            }
                            if !errored && let Some(reply) = parser.reply_state() {
                                if reply.reasoning.len() > emitted_reasoning
                                    && reply.reasoning.is_char_boundary(emitted_reasoning)
                                {
                                    let delta = reply.reasoning[emitted_reasoning..].to_string();
                                    emitted_reasoning = reply.reasoning.len();
                                    let _ = sender.send(Event::data(chunk_json_with_session(
                                        &id_str,
                                        &model_str,
                                        created,
                                        StreamDelta {
                                            reasoning_content: Some(delta),
                                            ..Default::default()
                                        },
                                        None,
                                        None,
                                        Some(&session_id_str),
                                    )));
                                }
                                if reply.text.len() > emitted && reply.text.is_char_boundary(emitted) {
                                    let delta = reply.text[emitted..].to_string();
                                    emitted = reply.text.len();
                                    let _ = sender.send(Event::data(chunk_json_with_session(
                                        &id_str,
                                        &model_str,
                                        created,
                                        StreamDelta {
                                            content: Some(delta),
                                            ..Default::default()
                                        },
                                        None,
                                        None,
                                        Some(&session_id_str),
                                    )));
                                }
                                // Candidates re-feed their full markup each round; the streamer
                                // diffs internally, exactly like the text chat path.
                                let finished_count = reply.tool_calls.len();
                                for (offset, raw) in reply.tool_call_candidates.iter().enumerate() {
                                    let index = finished_count + offset;
                                    let deltas = tool_call_streamers
                                        .entry(index)
                                        .or_insert_with(ToolCallStreamer::new)
                                        .update(index, raw, &parameter_types);
                                    match deltas {
                                        Ok(deltas) => {
                                            for delta in deltas {
                                                let _ = sender.send(Event::data(chunk_json_with_session(
                                                    &id_str,
                                                    &model_str,
                                                    created,
                                                    StreamDelta {
                                                        tool_calls: Some(vec![delta]),
                                                        ..Default::default()
                                                    },
                                                    None,
                                                    None,
                                                    Some(&session_id_str),
                                                )));
                                            }
                                        },
                                        Err(error) => {
                                            tracing::error!("VLM tool call stream error: {error}");
                                            errored = true;
                                        },
                                    }
                                }
                                while emitted_tool_calls < reply.tool_calls.len() {
                                    let index = emitted_tool_calls;
                                    emitted_tool_calls += 1;
                                    let call_value = &reply.tool_calls[index];
                                    let call = coerce_tool_call(
                                        &uzu::types::basic::ToolCall {
                                            identifier: None,
                                            name: call_value.name.clone(),
                                            arguments: uzu::types::basic::Value {
                                                json: call_value.arguments.clone(),
                                            },
                                        },
                                        &parameter_types,
                                    );
                                    let delta = match tool_call_streamers.remove(&index) {
                                        Some(mut streamer) => match streamer.finish(index, &call) {
                                            Ok(delta) => delta,
                                            Err(error) => {
                                                tracing::error!("VLM tool call stream error: {error}");
                                                errored = true;
                                                break;
                                            },
                                        },
                                        None => oai_tool_call(Some(index), &call),
                                    };
                                    let _ = sender.send(Event::data(chunk_json_with_session(
                                        &id_str,
                                        &model_str,
                                        created,
                                        StreamDelta {
                                            tool_calls: Some(vec![delta]),
                                            ..Default::default()
                                        },
                                        None,
                                        None,
                                        Some(&session_id_str),
                                    )));
                                }
                                if emitted_tool_calls > 0 {
                                    finish_reason = "tool_calls".to_string();
                                }
                            }
                        } else if let Ok(Some(chunk)) = decoder.step(curr_token as u32) {
                            let delta = if chunk.contains("</think>") {
                                in_thinking = false;
                                let parts: Vec<&str> = chunk.split("</think>").collect();
                                let reasoning = if !parts[0].is_empty() { Some(parts[0].to_string()) } else { None };
                                let content = if parts.len() > 1 && !parts[1].trim_start_matches('\n').is_empty() {
                                    Some(parts[1].trim_start_matches('\n').to_string())
                                } else {
                                    None
                                };
                                StreamDelta {
                                    content,
                                    reasoning_content: reasoning,
                                    ..StreamDelta::default()
                                }
                            } else if in_thinking {
                                StreamDelta {
                                    reasoning_content: Some(chunk),
                                    ..Default::default()
                                }
                            } else {
                                StreamDelta {
                                    content: Some(chunk),
                                    ..Default::default()
                                }
                            };

                            let sent = sender.send(Event::data(chunk_json_with_session(
                                &id_str,
                                &model_str,
                                created,
                                delta,
                                None,
                                None,
                                Some(&session_id_str),
                            )));
                            if sent.is_err() {
                                break;
                            }
                        }
                        tokens_count += 1;
                    },
                    Err(err) => {
                        tracing::error!("VLM decode step error: {err}");
                        break;
                    }
                }
            }

            if reply_parser.is_none()
                && let Ok(Some(tail)) = decoder.finish()
                && !tail.is_empty()
            {
                let delta = if in_thinking {
                    StreamDelta {
                        reasoning_content: Some(tail),
                        ..Default::default()
                    }
                } else {
                    StreamDelta {
                        content: Some(tail),
                        ..Default::default()
                    }
                };
                let _ = sender.send(Event::data(chunk_json_with_session(
                    &id_str,
                    &model_str,
                    created,
                    delta,
                    None,
                    None,
                    Some(&session_id_str),
                )));
            }

            let gen_u32: Vec<u32> = generated_tokens.iter().map(|&x| x as u32).collect();
            let full_text = lm.tokenizer().decode(&gen_u32, false).unwrap_or_default();

            // A clean-stop turn that ended inside the prefilled think section
            // (never any `</think>` in the generated text) put its answer there,
            // not reasoning; follow the streamed reasoning deltas with a content
            // delta carrying the body (see promote_unclosed_think).
            if !errored
                && reply_parser.is_none()
                && finish_reason == "stop"
                && in_thinking
                && !full_text.trim().is_empty()
                && !full_text.contains("</think>")
            {
                tracing::warn!(
                    "model ended turn inside an open <think> section; promoting think content to the reply body"
                );
                let _ = sender.send(Event::data(chunk_json_with_session(
                    &id_str,
                    &model_str,
                    created,
                    StreamDelta {
                        content: Some(full_text.trim().to_string()),
                        ..Default::default()
                    },
                    None,
                    None,
                    Some(&session_id_str),
                )));
            }

            let mut summary_text = full_text.clone();
            if !errored {
                if let Some(parser) = reply_parser.as_ref()
                    && let Some(reply) = parser.reply_state()
                {
                    // Flush any final text that grew since the last in-loop emission.
                    if reply.text.len() > emitted && reply.text.is_char_boundary(emitted) {
                        let delta = reply.text[emitted..].to_string();
                        let _ = sender.send(Event::data(chunk_json_with_session(
                            &id_str,
                            &model_str,
                            created,
                            StreamDelta {
                                content: Some(delta),
                                ..Default::default()
                            },
                            None,
                            None,
                            Some(&session_id_str),
                        )));
                    }
                    // Unfinished markup must never execute as a partial batch.
                    if !reply.tool_call_candidates.is_empty() {
                        finish_reason = "malformed_tool_call".to_string();
                    }
                    // A clean-stop turn that never closed the prefilled think
                    // section put its answer inside the think block; the streamed
                    // reasoning deltas already went out, so follow them with a
                    // content delta carrying the promoted body.
                    let (_reasoning, content, promoted) =
                        promote_unclosed_think(reply.reasoning.clone(), reply.text.clone(), &finish_reason, &full_text);
                    if promoted {
                        let _ = sender.send(Event::data(chunk_json_with_session(
                            &id_str,
                            &model_str,
                            created,
                            StreamDelta {
                                content: Some(content.clone()),
                                ..Default::default()
                            },
                            None,
                            None,
                            Some(&session_id_str),
                        )));
                    }
                    // The summary keeps the visible content only: clients echo assistant
                    // messages back without reasoning or tool-call markup, so prefix-cache
                    // matching stays exact.
                    summary_text = if promoted { content } else { reply.text.clone() };
                } else if reply_parser.is_some() {
                    finish_reason = "malformed_tool_call".to_string();
                }
            }

            let assistant_content = if summary_text.is_empty() {
                None
            } else {
                Some(summary_text)
            };
            if !errored {
                if let Some(tail) = tool_tail_summaries {
                    session.history.extend(tail);
                    session.history.push(crate::server::vlm_session::SessionMessageSummary {
                        role: "assistant".to_string(),
                        content: assistant_content,
                        image_hashes: vec![],
                        video_hashes: vec![],
                    });
                    session.prompt_tokens_count += prompt_tokens_count + tokens_count;
                } else if let Some(query) = followup_query {
                    session.history.push(crate::server::vlm_session::SessionMessageSummary {
                        role: "user".to_string(),
                        content: Some(query),
                        image_hashes: vec![],
                        video_hashes: vec![],
                    });
                    session.history.push(crate::server::vlm_session::SessionMessageSummary {
                        role: "assistant".to_string(),
                        content: assistant_content,
                        image_hashes: vec![],
                        video_hashes: vec![],
                    });
                    session.prompt_tokens_count += prompt_tokens_count + tokens_count;
                } else {
                    session.history = incoming_summary;
                    session.history.push(crate::server::vlm_session::SessionMessageSummary {
                        role: "assistant".to_string(),
                        content: assistant_content,
                        image_hashes: vec![],
                        video_hashes: vec![],
                    });
                    session.prompt_tokens_count = prompt_tokens_count + tokens_count;
                }
                session.last_generation_tool_call = emitted_tool_calls > 0;
            }
            drop(session);

            let usage = ChatCompletionUsage {
                prompt_tokens: cached_tokens + prompt_tokens_count as u32,
                completion_tokens: tokens_count as u32,
                total_tokens: cached_tokens + prompt_tokens_count as u32 + tokens_count as u32,
                prompt_tokens_details: Some(ChatCompletionPromptTokensDetails {
                    cached_tokens,
                }),
                spec_verify_ct: None,
            };

            if errored {
                let body = oai_error_body(Status::InternalServerError, "backend_error", "VLM reply parsing failed");
                let _ = sender.send(Event::data(serde_json::to_string(&body).unwrap_or_default()));
            } else {
                let _ = sender.send(Event::data(chunk_json_with_session(
                    &id_str,
                    &model_str,
                    created,
                    StreamDelta::default(),
                    Some(finish_reason.clone()),
                    Some(usage),
                    Some(&session_id_str),
                )));
            }
            let _ = sender.send(Event::data("[DONE]"));

            let vlm_stats = ChatReplyStats {
                tokens_count_input: Some(cached_tokens + prompt_tokens_count as u32),
                tokens_count_input_cached: Some(cached_tokens),
                tokens_count_output: Some(tokens_count as u32),
                ..Default::default()
            };
            if errored {
                log.fail("VLM reply parsing failed");
            } else {
                log.finish(&finish_reason, Some(&vlm_stats), Vec::new());
            }
        });

        let body: Pin<Box<dyn Stream<Item = Event> + Send>> = Box::pin(UnboundedReceiverStream::new(receiver));
        ChatCompletionResult::stream_with_session(EventStream::from(body), active_session_id)
    } else {
        let mut generated = vec![first_token];
        let mut curr_token = first_token;

        while generated.len() < max_tokens && !stop_token_ids.contains(&curr_token) {
            match tokio::task::block_in_place(|| {
                vlm.decode_step_with_state(&lm, &mut session.state, curr_token, &mut sampler)
            }) {
                Ok(next_tok) => {
                    curr_token = next_tok;
                    if stop_token_ids.contains(&curr_token) {
                        break;
                    }
                    generated.push(curr_token);
                },
                Err(err) => {
                    tracing::error!("VLM decode step error: {err}");
                    break;
                }
            }
        }

        let gen_u32: Vec<u32> = generated.iter().map(|&x| x as u32).collect();
        let full_text = lm.tokenizer().decode(&gen_u32, false).unwrap_or_default();

        // Tool requests classify the reply through the engine's parsing stack; plain
        // requests keep the think-split below.
        let (finish_reason, response_tool_calls, reasoning_content, content, summary_text) =
            if let Some(parser) = reply_parser.as_mut() {
                let mut parse_ok = true;
                for token_id in &gen_u32 {
                    if let Err(error) = parser.push_sampled_token(*token_id) {
                        tracing::error!("VLM reply parser error: {error}");
                        parse_ok = false;
                        break;
                    }
                }
                let reply = if parse_ok {
                    parser.reply_state()
                } else {
                    None
                };
                match reply {
                    Some(reply) => {
                        let mut finish_reason = if reply.tool_calls.is_empty() {
                            "stop"
                        } else {
                            "tool_calls"
                        };
                        let tool_calls = if reply.tool_calls.is_empty() {
                            None
                        } else {
                            Some(
                                reply
                                    .tool_calls
                                    .iter()
                                    .enumerate()
                                    .map(|(index, call_value)| {
                                        oai_tool_call(
                                            Some(index),
                                            &coerce_tool_call(
                                                &uzu::types::basic::ToolCall {
                                                    identifier: None,
                                                    name: call_value.name.clone(),
                                                    arguments: uzu::types::basic::Value {
                                                        json: call_value.arguments.clone(),
                                                    },
                                                },
                                                &parameter_types,
                                            ),
                                        )
                                    })
                                    .collect(),
                            )
                        };
                        // Unfinished markup must never execute as a partial batch.
                        if !reply.tool_call_candidates.is_empty() {
                            finish_reason = "malformed_tool_call";
                        }
                        // A clean-stop turn that never closed the prefilled think
                        // section is the model putting its answer inside the think
                        // block; promote it to the body (see promote_unclosed_think).
                        let (reasoning, content, _promoted) = promote_unclosed_think(
                            reply.reasoning.clone(),
                            reply.text.clone(),
                            finish_reason,
                            &full_text,
                        );
                        let summary = content.clone();
                        (finish_reason.to_string(), tool_calls, reasoning, content, summary)
                    },
                    None => ("stop".to_string(), None, None, full_text.clone(), full_text),
                }
            } else if enable_thinking {
                // The prompt supplied the opening `<think>`; everything before
                // `</think>` is reasoning, and a truncated turn is all reasoning.
                // An unclosed section on a clean stop is the model falling off
                // the template contract — promote it to the reply body instead.
                let (reasoning, body) = match full_text.split_once("</think>") {
                    Some((reasoning, body)) => {
                        let reasoning = reasoning.trim();
                        (
                            (!reasoning.is_empty()).then(|| reasoning.to_string()),
                            body.trim_start_matches('\n').to_string(),
                        )
                    },
                    None => {
                        let reasoning = full_text.trim().to_string();
                        if reasoning.is_empty() {
                            (None, String::new())
                        } else {
                            let (reasoning, body, _promoted) =
                                promote_unclosed_think(reasoning, String::new(), "stop", &full_text);
                            (reasoning, body)
                        }
                    },
                };
                ("stop".to_string(), None, reasoning, body.clone(), body)
            } else {
                let (reasoning, body) = split_think_text(&full_text);
                ("stop".to_string(), None, reasoning, body.clone(), body)
            };

        let response = ChatCompletionResponse {
            id: id.to_string(),
            object: "chat.completion".to_string(),
            created,
            model: model.to_string(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: OaiMessage {
                    role: "assistant".to_string(),
                    content: Some(content),
                    reasoning_content,
                    tool_calls: response_tool_calls.clone(),
                    tool_call_id: None,
                    audio: None,
                    image_urls: Vec::new(),
                    images: Vec::new(),
                    video_urls: Vec::new(),
                    videos: Vec::new(),
                    audios: Vec::new(),
                },
                finish_reason: finish_reason.clone(),
            }],
            usage: ChatCompletionUsage {
                prompt_tokens: cached_tokens + prompt_tokens_count as u32,
                completion_tokens: generated.len() as u32,
                total_tokens: cached_tokens + prompt_tokens_count as u32 + generated.len() as u32,
                prompt_tokens_details: Some(ChatCompletionPromptTokensDetails {
                    cached_tokens,
                }),
                spec_verify_ct: None,
            },
            session_id: Some(active_session_id.clone()),
        };

        let assistant_content = if summary_text.is_empty() {
            None
        } else {
            Some(summary_text)
        };
        if let Some(tail) = tool_tail_summaries {
            session.history.extend(tail);
            session.history.push(crate::server::vlm_session::SessionMessageSummary {
                role: "assistant".to_string(),
                content: assistant_content,
                image_hashes: vec![],
                video_hashes: vec![],
            });
            session.prompt_tokens_count += prompt_tokens_count + generated.len();
        } else if let Some(query) = followup_query {
            session.history.push(crate::server::vlm_session::SessionMessageSummary {
                role: "user".to_string(),
                content: Some(query),
                image_hashes: vec![],
                video_hashes: vec![],
            });
            session.history.push(crate::server::vlm_session::SessionMessageSummary {
                role: "assistant".to_string(),
                content: assistant_content,
                image_hashes: vec![],
                video_hashes: vec![],
            });
            session.prompt_tokens_count += prompt_tokens_count + generated.len();
        } else {
            session.history = incoming_summary;
            session.history.push(crate::server::vlm_session::SessionMessageSummary {
                role: "assistant".to_string(),
                content: assistant_content,
                image_hashes: vec![],
                video_hashes: vec![],
            });
            session.prompt_tokens_count = prompt_tokens_count + generated.len();
        }
        session.last_generation_tool_call = response_tool_calls.is_some();
        drop(session);

        let vlm_stats = ChatReplyStats {
            tokens_count_input: Some(cached_tokens + prompt_tokens_count as u32),
            tokens_count_input_cached: Some(cached_tokens),
            tokens_count_output: Some(generated.len() as u32),
            ..Default::default()
        };
        log.finish(&finish_reason, Some(&vlm_stats), Vec::new());

        let body = serde_json::to_vec(&response).unwrap_or_default();
        let stream: Pin<Box<dyn Stream<Item = Vec<u8>> + Send>> = Box::pin(futures_util::stream::once(async move { body }));
        ChatCompletionResult::json_with_session(Status::Ok, ByteStream::from(stream), active_session_id)
    }
}

#[allow(private_interfaces)]
#[post("/chat/completions", format = "json", data = "<body>")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_chat_completions(
    body: Data<'_>,
    state: &State<ServerState>,
    chat: &State<ChatState>,
    request_info: &RequestInfo,
    content_type: &ContentType,
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
    let log_body = if content_type.is_json() {
        serde_json::from_str::<serde_json::Value>(&body).map(|value| value.to_string()).unwrap_or_else(|_| body.clone())
    } else {
        body.clone()
    };
    tracing::debug!("--> {} {} body={}", request_info.method, request_info.uri, log_body);

    let mut request = match serde_json::from_str::<ChatCompletionRequest>(&body) {
        Ok(request) => request,
        Err(error) => {
            tracing::info!("rejected: {error}");
            return invalid_request_response(
                "body",
                "invalid_request",
                format!("failed to parse chat completion request: {error}"),
            );
        },
    };

    if let Err(err) = resolve_message_images(&mut request.messages).await {
        return invalid_request_response("image_url", "invalid_request_error", err);
    }

    let created = request_info.created_at;
    let is_stream = request.stream.unwrap_or(false);
    let id = request_info.id.as_str().to_owned();
    // The VLM path treats an unspecified reasoning effort as thinking off; reflect the
    // resolved value in the request log instead of "unspecified".
    let enable_thinking = match parse_enable_thinking(&request) {
        Ok(Some(v)) => v,
        _ => request
            .reasoning_effort
            .as_ref()
            .and_then(|v| v.as_str())
            .map(|s| s != "none")
            .unwrap_or(false),
    };
    let effort_display = request
        .reasoning_effort
        .as_ref()
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| if enable_thinking { "on (default)".to_string() } else { "off (default)".to_string() });

    // Slot gate: the request needs a resident model matching its `model`
    // field. Snapshot (not lock) — in-flight requests keep their Arc alive
    // across a swap, so no lock is held while generating.
    let requested_model = request.model.clone();
    let loaded = match chat.slot.snapshot().await {
        SlotSnapshot::Ready(loaded) => match &requested_model {
            // A missing model field is accepted and served by the resident model.
            Some(model) if *model != loaded.model_name => {
                let message = format!(
                    "requested model '{model}' is not resident (current: '{}')",
                    loaded.model_name
                );
                let log = RequestLog::start(
                    &request,
                    model.as_str(),
                    is_stream,
                    effort_display.as_str(),
                    request_info,
                    &state.access_log,
                );
                log.fail(&message);
                return model_not_loaded_response(Some(model), Some(&loaded.model_name), &message);
            },
            _ => loaded,
        },
        other => {
            let detail = match &other {
                SlotSnapshot::Loading { model, .. } => format!("model '{model}' is loading"),
                SlotSnapshot::Failed { model, error, .. } => format!("model '{model}' failed to load: {error}"),
                _ => "no model is loaded".to_string(),
            };
            let log = RequestLog::start(
                &request,
                requested_model.as_deref().unwrap_or("(none)"),
                is_stream,
                effort_display.as_str(),
                request_info,
                &state.access_log,
            );
            log.fail(&detail);
            return model_not_loaded_response(requested_model.as_deref(), other.model_name(), &detail);
        },
    };

    let model = loaded.model_name.clone();
    let log = RequestLog::start(&request, &model, is_stream, effort_display.as_str(), request_info, &state.access_log);

    let is_omni = loaded.audio.is_some()
        && loaded
            .vlm_pipeline
            .as_ref()
            .is_some_and(|vlm| matches!(vlm.as_ref(), uzu_engine::vision::AnyVlmPipeline::Lfm3(_)));

    if loaded.vlm_pipeline.is_some() && !is_omni {
        return handle_vlm_completion(
            &request,
            &loaded,
            &model,
            created,
            &id,
            is_stream,
            enable_thinking,
            log,
            request_info,
        )
        .await;
    }

    let has_audio_input = request.messages.iter().any(|m| !m.audios.is_empty());
    let wants_audio_output = request
        .modalities
        .as_ref()
        .is_some_and(|ms| ms.iter().any(|m| m == "audio"));
    if loaded.audio.is_some() || has_audio_input || wants_audio_output {
        let requested_session_id = request
            .session_id
            .as_deref()
            .or(request_info.session_id.as_deref());
        return handle_audio_chat_completion(
            &request,
            &loaded,
            &model,
            created,
            &id,
            is_stream,
            enable_thinking,
            wants_audio_output,
            requested_session_id,
            log,
        )
        .await;
    }

    if let Err((param, message)) = validate_sampling(&request) {
        log.fail(&message);
        return invalid_request_response(param, "out_of_range", message);
    }
    let config = match build_reply_config(&request) {
        Ok(config) => config,
        Err(error) => {
            log.fail(&error.message());
            return invalid_request_response("response_format", error.code(), error.message());
        },
    };
    let messages = match build_messages(&request, loaded.thinking_support) {
        Ok(messages) => messages,
        Err(error) => {
            let param = error.param();
            let code = error.code();
            let detail = error.into_detail();
            log.fail(&detail);
            return invalid_request_response(param, code, detail);
        },
    };

    let parameter_types = tool_parameter_types(request.tools.as_deref());

    if is_stream {
        let session = Arc::clone(&loaded.session);
        let (sender, receiver) = mpsc::unbounded_channel::<Event>();
        tokio::spawn(
            run_stream(session, messages, config, id, model, created, chat.prefix_cache, parameter_types, sender, log)
                .in_current_span(),
        );
        let body: Pin<Box<dyn Stream<Item = Event> + Send>> = Box::pin(UnboundedReceiverStream::new(receiver));
        ChatCompletionResult::stream(EventStream::from(body))
    } else {
        let session = Arc::clone(&loaded.session);
        let (sender, receiver) = mpsc::unbounded_channel::<Vec<u8>>();
        tokio::spawn(
            run_blocking(
                session,
                messages,
                config,
                id,
                model,
                created,
                chat.prefix_cache,
                parameter_types,
                sender,
                log,
            )
            .in_current_span(),
        );
        let body: Pin<Box<dyn Stream<Item = Vec<u8>> + Send>> = Box::pin(UnboundedReceiverStream::new(receiver));
        // Hand the body to Rocket immediately so it can write keepalives and
        // drop the receiver when the client disconnects. Any backend error
        // occurs after the 200 response is committed and is sent in the body.
        ChatCompletionResult::json(Status::Ok, ByteStream::from(body))
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_audio_chat_completion(
    request: &ChatCompletionRequest,
    loaded: &LoadedModel,
    model: &str,
    created: i64,
    id: &str,
    is_stream: bool,
    enable_thinking: bool,
    wants_audio_output: bool,
    requested_session_id: Option<&str>,
    log: RequestLog,
) -> ChatCompletionResult {
    let Some(audio) = loaded.audio.as_ref().map(Arc::clone) else {
        log.fail("audio runtime unavailable");
        return invalid_request_response(
            "model",
            "invalid_request_error",
            "resident model does not support audio".to_string(),
        );
    };
    let chosen_tools = match choose_tools(request.tools.as_deref(), request.tool_choice.as_ref()) {
        Ok(tools) => tools,
        Err(err) => {
            log.fail(&err);
            return invalid_request_response("tool_choice", "invalid_tool_choice", err);
        },
    };
    let has_tools = !chosen_tools.is_empty();
    let parameter_types = tool_parameter_types(request.tools.as_deref());

    let mut turns: Vec<crate::common::audio::AudioInputTurn> = Vec::with_capacity(request.messages.len());
    for m in &request.messages {
        let mut text = m.content.clone().unwrap_or_default();
        if m.role == "tool" {
            text = format!("<tool_response>{text}</tool_response>");
        } else if m.role == "assistant"
            && let Some(calls) = m.tool_calls.as_deref().filter(|c| !c.is_empty()) {
                let call_str = format_lfm3_audio_assistant_tool_calls(calls);
                if text.is_empty() {
                    text = call_str;
                } else {
                    text = format!("{text}\n{call_str}");
                }
            }
        if m.audios.len() <= 1 {
            turns.push(crate::common::audio::AudioInputTurn {
                role: m.role.clone(),
                text,
                wav_bytes: m.audios.first().cloned(),
            });
        } else {
            for (idx, wav) in m.audios.iter().enumerate() {
                turns.push(crate::common::audio::AudioInputTurn {
                    role: m.role.clone(),
                    text: if idx + 1 == m.audios.len() {
                        text.clone()
                    } else {
                        String::new()
                    },
                    wav_bytes: Some(wav.clone()),
                });
            }
        }
    }
    let requested_sid_owned = requested_session_id.map(str::to_string);
    let fallback_sid = format!("aud_{}", &id[..id.len().min(12)]);
    let active_session_id = requested_sid_owned.clone().unwrap_or_else(|| fallback_sid.clone());
    let is_delta_followup = requested_session_id.is_some()
        && turns.iter().all(|t| t.role == "user" || t.role == "tool")
        && audio.lock().await.is_active_session(requested_session_id);

    if has_tools && !is_delta_followup {
        let preamble = format_lfm3_audio_tool_preamble(&chosen_tools);
        if let Some(sys) = turns.iter_mut().find(|t| t.role == "system") {
            if sys.text.is_empty() {
                sys.text = preamble;
            } else {
                sys.text = format!("{}\n\n{preamble}", sys.text);
            }
        } else {
            turns.insert(
                0,
                crate::common::audio::AudioInputTurn {
                    role: "system".to_string(),
                    text: preamble,
                    wav_bytes: None,
                },
            );
        }
    }
    let wants_text_output = request
        .modalities
        .as_ref()
        .is_none_or(|ms| ms.iter().any(|m| m == "text"));

    let mode = match (wants_audio_output, wants_text_output) {
        (true, true) => Some(uzu_engine::audio::lfm3_audio::pipeline::GenerationMode::Interleaved),
        (true, false) => Some(uzu_engine::audio::lfm3_audio::pipeline::GenerationMode::Audio),
        (false, _) => Some(uzu_engine::audio::lfm3_audio::pipeline::GenerationMode::Text),
    };
    let assistant_prefill = if mode == Some(uzu_engine::audio::lfm3_audio::pipeline::GenerationMode::Text) && !has_tools {
        match request.reasoning_effort.as_ref().and_then(serde_json::Value::as_str) {
            Some("none") | Some("off") => Some("<think></think>".to_string()),
            Some("low") | Some("medium") | Some("high") | Some("on") => Some("<think>".to_string()),
            _ => None,
        }
    } else {
        None
    };
    let audio_options = crate::common::audio::AudioGenerateOptions {
        audio_temperature: request.audio_temperature,
        audio_top_k: request.audio_top_k,
        audio_repetition_penalty: request.audio_repetition_penalty,
        audio_repetition_window: request.audio_repetition_window,
        assistant_prefill,
    };
    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(512) as usize;
    let temperature = request.temperature.map(|t| t as f32);
    let seed = request.seed;
    let lm = Arc::clone(&loaded.lm);

    // If an Lfm3 vision pipeline is also attached, prepare OmniMessage list (images + videos + audios + text).
    let omni_vlm = loaded.vlm_pipeline.clone().filter(|v| {
        matches!(v.as_ref(), uzu_engine::vision::AnyVlmPipeline::Lfm3(_))
    });
    let omni_messages: Option<Vec<uzu_engine::omni::OmniMessage>> = if omni_vlm.is_some() {
        let mut msgs = Vec::with_capacity(request.messages.len());
        for m in &request.messages {
            let mut decoded_audios = Vec::with_capacity(m.audios.len());
            for wav in &m.audios {
                match crate::common::audio::decode_audio_bytes_16khz_mono(wav, None, None) {
                    Ok(samples) => decoded_audios.push(samples),
                    Err(err) => {
                        log.fail(&err.to_string());
                        return invalid_request_response("messages", "invalid_request_error", err.to_string());
                    },
                }
            }
            let tool_calls = m
                .tool_calls
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .map(|c| uzu_engine::vision::MultimodalToolCall {
                    name: c.function.name.clone(),
                    arguments: c.function.arguments.clone(),
                })
                .collect();
            msgs.push(uzu_engine::omni::OmniMessage {
                role: m.role.clone(),
                content: m.content.clone(),
                reasoning_content: m.reasoning_content.clone(),
                tool_calls,
                images: m.images.iter().map(|img| img.image.clone()).collect(),
                videos: m
                    .videos
                    .iter()
                    .map(|frames| uzu_engine::omni::OmniVideoInput {
                        frames: frames.clone(),
                        fps: crate::common::video::VIDEO_FPS,
                        frame_indices: None,
                    })
                    .collect(),
                audios: decoded_audios,
                audio_out_codes: Vec::new(),
            });
        }
        Some(msgs)
    } else {
        None
    };
    let omni_tools_preamble = if omni_vlm.is_some() && has_tools {
        Some(format_lfm3_audio_tool_preamble(&chosen_tools))
    } else {
        None
    };

    if is_stream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let id_owned = id.to_string();
        let model_owned = model.to_string();
        let _ = tx.send(Event::data(chunk_json(
            &id_owned,
            &model_owned,
            created,
            StreamDelta {
                role: Some("assistant".to_string()),
                ..StreamDelta::default()
            },
            None,
            None,
        )));

        let session_for_stream = active_session_id.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = audio.blocking_lock();
            let mut streamed_audio_chunks = 0usize;
            let mut on_text = |delta: &str| {
                if !delta.is_empty() {
                    let _ = tx.send(Event::data(chunk_json(
                        &id_owned,
                        &model_owned,
                        created,
                        StreamDelta {
                            content: Some(delta.to_string()),
                            ..StreamDelta::default()
                        },
                        None,
                        None,
                    )));
                }
            };
            let mut on_wav = |wav_chunk: &[u8]| {
                streamed_audio_chunks += 1;
                let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, wav_chunk);
                let _ = tx.send(Event::data(chunk_json(
                    &id_owned,
                    &model_owned,
                    created,
                    StreamDelta {
                        audio: Some(OaiAudioOutput {
                            id: format!("audio_{id_owned}_{streamed_audio_chunks}"),
                            data: b64,
                            transcript: String::new(),
                        }),
                        ..StreamDelta::default()
                    },
                    None,
                    None,
                )));
            };
            let res = if let (Some(vlm), Some(msgs)) = (omni_vlm.as_ref(), omni_messages.as_ref()) {
                let uzu_engine::vision::AnyVlmPipeline::Lfm3(vision) = vlm.as_ref() else {
                    unreachable!();
                };
                guard
                    .generate_omni_chat_with_options_streaming(
                        &lm,
                        vision,
                        msgs,
                        omni_tools_preamble.as_deref(),
                        enable_thinking,
                        mode,
                        wants_audio_output,
                        max_tokens,
                        temperature,
                        seed,
                        &audio_options,
                        on_text,
                        on_wav,
                    )
                    .map(|out| (out, false))
            } else {
                guard.generate_chat_with_options_streaming(
                    &lm,
                    requested_sid_owned.as_deref(),
                    fallback_sid,
                    &turns,
                    mode,
                    wants_audio_output,
                    max_tokens,
                    temperature,
                    seed,
                    &audio_options,
                    &mut on_text,
                    &mut on_wav,
                )
            };
            match res {
                Ok((out, _kv_reused)) => {
                    let frame_count = out.frames.len();
                    let completion_tokens = out.tokens.len() as u32;
                    let (tool_calls, clean_text) = if has_tools {
                        parse_lfm3_audio_tool_calls(&out.raw_text, &parameter_types)
                    } else {
                        (None, out.text.clone())
                    };
                    let finish_reason = if tool_calls.is_some() { "tool_calls" } else { "stop" };
                    if let Some(reasoning) = out.reasoning.as_deref().filter(|r| !r.is_empty()) {
                        let _ = tx.send(Event::data(chunk_json(
                            &id_owned,
                            &model_owned,
                            created,
                            StreamDelta {
                                reasoning_content: Some(reasoning.to_string()),
                                ..StreamDelta::default()
                            },
                            None,
                            None,
                        )));
                    }
                    if let Some(calls) = tool_calls {
                        let _ = tx.send(Event::data(chunk_json(
                            &id_owned,
                            &model_owned,
                            created,
                            StreamDelta {
                                tool_calls: Some(calls),
                                ..StreamDelta::default()
                            },
                            None,
                            None,
                        )));
                    } else if clean_text.is_empty() && frame_count > 0 {
                        let _ = tx.send(Event::data(chunk_json(
                            &id_owned,
                            &model_owned,
                            created,
                            StreamDelta {
                                content: Some(format!("[Speech reply · {frame_count} frames]")),
                                ..StreamDelta::default()
                            },
                            None,
                            None,
                        )));
                    }
                    if streamed_audio_chunks == 0
                        && let Some(wav) = out.wav_bytes {
                            let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &wav);
                            let _ = tx.send(Event::data(chunk_json(
                                &id_owned,
                                &model_owned,
                                created,
                                StreamDelta {
                                    audio: Some(OaiAudioOutput {
                                        id: format!("audio_{id_owned}"),
                                        data: b64,
                                        transcript: clean_text,
                                    }),
                                    ..StreamDelta::default()
                                },
                                None,
                                None,
                            )));
                        }
                    let _ = tx.send(Event::data(chunk_json(
                        &id_owned,
                        &model_owned,
                        created,
                        StreamDelta::default(),
                        Some(finish_reason.to_string()),
                        Some(ChatCompletionUsage {
                            prompt_tokens: 0,
                            completion_tokens,
                            total_tokens: completion_tokens,
                            prompt_tokens_details: None,
                            spec_verify_ct: None,
                        }),
                    )));
                    let _ = tx.send(Event::data("[DONE]"));
                },
                Err(err) => {
                    log.fail(&err.to_string());
                    let err_json = serde_json::json!({
                        "error": { "message": err.to_string(), "type": "server_error" }
                    });
                    let _ = tx.send(Event::data(err_json.to_string()));
                    let _ = tx.send(Event::data("[DONE]"));
                },
            }
        });

        let body: Pin<Box<dyn Stream<Item = Event> + Send>> =
            Box::pin(futures_util::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|ev| (ev, rx))
            }));
        return ChatCompletionResult::stream_with_session(EventStream::from(body), session_for_stream);
    }

    let res = tokio::task::spawn_blocking(move || {
        let mut guard = audio.blocking_lock();
        if let (Some(vlm), Some(msgs)) = (omni_vlm.as_ref(), omni_messages.as_ref()) {
            let uzu_engine::vision::AnyVlmPipeline::Lfm3(vision) = vlm.as_ref() else {
                unreachable!();
            };
            guard
                .generate_omni_chat_with_options_streaming(
                    &lm,
                    vision,
                    msgs,
                    omni_tools_preamble.as_deref(),
                    enable_thinking,
                    mode,
                    wants_audio_output,
                    max_tokens,
                    temperature,
                    seed,
                    &audio_options,
                    |_| {},
                    |_| {},
                )
                .map(|out| (out, false))
        } else {
            guard.generate_chat_with_options_streaming(
                &lm,
                requested_sid_owned.as_deref(),
                fallback_sid,
                &turns,
                mode,
                wants_audio_output,
                max_tokens,
                temperature,
                seed,
                &audio_options,
                |_| {},
                |_| {},
            )
        }
    })
    .await;

    let (out, _kv_reused) = match res {
        Ok(Ok(out)) => out,
        Ok(Err(err)) => {
            log.fail(&err.to_string());
            return invalid_request_response("messages", "invalid_request_error", err.to_string());
        },
        Err(err) => {
            log.fail(&err.to_string());
            return invalid_request_response("messages", "server_error", err.to_string());
        },
    };

    let frame_count = out.frames.len();
    let (tool_calls, clean_text) = if has_tools {
        parse_lfm3_audio_tool_calls(&out.raw_text, &parameter_types)
    } else {
        (None, out.text.clone())
    };
    let finish_reason = if tool_calls.is_some() { "tool_calls" } else { "stop" };
    let audio_field = out.wav_bytes.map(|wav| OaiAudioOutput {
        id: format!("audio_{id}"),
        data: base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &wav),
        transcript: clean_text.clone(),
    });
    let display_content = if tool_calls.is_some() {
        (!clean_text.is_empty()).then_some(clean_text)
    } else if clean_text.is_empty() && frame_count > 0 {
        Some(format!("[Speech reply · {frame_count} frames]"))
    } else {
        Some(clean_text)
    };
    let completion_tokens = out.tokens.len() as u32;
    let response = ChatCompletionResponse {
        id: id.to_string(),
        object: "chat.completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: OaiMessage {
                role: "assistant".to_string(),
                content: display_content,
                reasoning_content: out.reasoning,
                tool_calls,
                tool_call_id: None,
                audio: audio_field,
                image_urls: Vec::new(),
                images: Vec::new(),
                video_urls: Vec::new(),
                videos: Vec::new(),
                audios: Vec::new(),
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage: ChatCompletionUsage {
            prompt_tokens: 0,
            completion_tokens,
            total_tokens: completion_tokens,
            prompt_tokens_details: None,
            spec_verify_ct: None,
        },
        session_id: Some(active_session_id.clone()),
    };
    let bytes = serde_json::to_vec(&response).unwrap_or_default();
    let stream_body: Pin<Box<dyn Stream<Item = Vec<u8>> + Send>> =
        Box::pin(futures_util::stream::once(async move { bytes }));
    ChatCompletionResult::json_with_session(Status::Ok, ByteStream::from(stream_body), active_session_id)
}

#[cfg(test)]
#[path = "../../unit/server/chat_completions_test.rs"]
mod tests;
