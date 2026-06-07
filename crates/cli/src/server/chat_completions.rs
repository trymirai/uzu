use std::{
    pin::Pin,
    str::FromStr,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use rocket::{
    Request, State,
    futures::Stream,
    post,
    response::{
        Responder,
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
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
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
    #[serde(default)]
    pub strict: Option<bool>,
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

pub enum ChatCompletionResult {
    Json(Json<ChatCompletionResponse>),
    Stream(EventStream<Pin<Box<dyn Stream<Item = Event> + Send>>>),
}

impl<'r> Responder<'r, 'r> for ChatCompletionResult {
    fn respond_to(
        self,
        request: &'r Request<'_>,
    ) -> rocket::response::Result<'r> {
        match self {
            ChatCompletionResult::Json(json) => json.respond_to(request),
            ChatCompletionResult::Stream(stream) => stream.respond_to(request),
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
    NonStrictJsonSchemaUnsupported,
    JsonSchemaSerializationFailed,
}

const JSON_OBJECT_SCHEMA: &str = r#"{"type":"object"}"#;

impl ResponseFormatError {
    fn message(&self) -> &'static str {
        match self {
            ResponseFormatError::GrammarUnsupported => {
                "response_format with JSON constraints requires building mirai server with capability-grammar"
            },
            ResponseFormatError::NonStrictJsonSchemaUnsupported => {
                "response_format json_schema requires json_schema.strict=true; non-strict JSON schemas are not supported"
            },
            ResponseFormatError::JsonSchemaSerializationFailed => {
                "response_format json_schema.schema could not be serialized"
            },
        }
    }
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

    config = match &request.response_format {
        Some(ResponseFormat::JsonObject) => with_response_format_grammar(
            config,
            Grammar::JsonSchema {
                schema: JSON_OBJECT_SCHEMA.to_string(),
            },
        )?,
        Some(ResponseFormat::JsonSchema {
            json_schema,
        }) => {
            if json_schema.strict != Some(true) {
                return Err(ResponseFormatError::NonStrictJsonSchemaUnsupported);
            }

            let schema = serde_json::to_string(&json_schema.schema)
                .map_err(|_| ResponseFormatError::JsonSchemaSerializationFailed)?;
            with_response_format_grammar(
                config,
                Grammar::JsonSchema {
                    schema,
                },
            )?
        },
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

fn stream_error_response(
    id: String,
    model: String,
    created: i64,
    message: &str,
) -> ChatCompletionResult {
    let (sender, receiver) = mpsc::unbounded_channel::<Event>();
    let _ = sender.send(Event::data(chunk_json(
        &id,
        &model,
        created,
        StreamDelta {
            role: None,
            content: Some(format!("Error: {message}")),
        },
        Some("stop".to_string()),
        None,
    )));
    let _ = sender.send(Event::data("[DONE]"));
    let body: Pin<Box<dyn Stream<Item = Event> + Send>> = Box::pin(UnboundedReceiverStream::new(receiver));
    ChatCompletionResult::Stream(EventStream::from(body))
}

fn response_format_error_response(
    id: String,
    model: String,
    created: i64,
    is_stream: bool,
    error: ResponseFormatError,
) -> ChatCompletionResult {
    let message = error.message();
    if is_stream {
        stream_error_response(id, model, created, message)
    } else {
        ChatCompletionResult::Json(Json(error_response(id, model, created, message)))
    }
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
        Err(error) => return response_format_error_response(id, model, created, is_stream, error),
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

    fn reply_config_error(json: &str) -> ResponseFormatError {
        build_reply_config(&request(json)).expect_err("invalid reply config")
    }

    #[test]
    fn response_format_maps_to_grammar() {
        // Absent response_format leaves generation unconstrained.
        assert!(reply_config(r#"{"messages":[]}"#).grammar.is_none());

        // "text" is the OpenAI default: also unconstrained.
        assert!(reply_config(r#"{"messages":[],"response_format":{"type":"text"}}"#).grammar.is_none());

        #[cfg(feature = "capability-grammar")]
        {
            // "json_object" constrains to a top-level JSON object.
            assert_eq!(
                reply_config(r#"{"messages":[],"response_format":{"type":"json_object"}}"#).grammar,
                Some(Grammar::JsonSchema {
                    schema: JSON_OBJECT_SCHEMA.to_string(),
                })
            );

            // Strict "json_schema" forwards the schema verbatim as a JSON string.
            let config = reply_config(
                r#"{"messages":[],"response_format":{"type":"json_schema","json_schema":{"name":"p","strict":true,"schema":{"type":"object"}}}}"#,
            );
            assert_eq!(
                config.grammar,
                Some(Grammar::JsonSchema {
                    schema: r#"{"type":"object"}"#.to_string(),
                })
            );
        }
    }

    #[test]
    fn response_format_rejects_grammar_without_capability() {
        #[cfg(not(feature = "capability-grammar"))]
        {
            assert_eq!(
                reply_config_error(r#"{"messages":[],"response_format":{"type":"json_object"}}"#),
                ResponseFormatError::GrammarUnsupported
            );
            assert_eq!(
                reply_config_error(
                    r#"{"messages":[],"response_format":{"type":"json_schema","json_schema":{"strict":true,"schema":{"type":"object"}}}}"#
                ),
                ResponseFormatError::GrammarUnsupported
            );
        }
    }

    #[test]
    fn response_format_rejects_non_strict_json_schema() {
        assert_eq!(
            reply_config_error(
                r#"{"messages":[],"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}}"#
            ),
            ResponseFormatError::NonStrictJsonSchemaUnsupported
        );
        assert_eq!(
            reply_config_error(
                r#"{"messages":[],"response_format":{"type":"json_schema","json_schema":{"strict":false,"schema":{"type":"object"}}}}"#
            ),
            ResponseFormatError::NonStrictJsonSchemaUnsupported
        );
    }

    #[test]
    fn response_format_errors_match_requested_response_shape() {
        match response_format_error_response(
            "chatcmpl-test".to_string(),
            "model".to_string(),
            0,
            false,
            ResponseFormatError::GrammarUnsupported,
        ) {
            ChatCompletionResult::Json(_) => {},
            ChatCompletionResult::Stream(_) => panic!("non-stream errors should return JSON responses"),
        }

        match response_format_error_response(
            "chatcmpl-test".to_string(),
            "model".to_string(),
            0,
            true,
            ResponseFormatError::GrammarUnsupported,
        ) {
            ChatCompletionResult::Stream(_) => {},
            ChatCompletionResult::Json(_) => panic!("stream errors should return streaming responses"),
        }
    }

    #[cfg(feature = "capability-grammar")]
    #[test]
    fn response_format_composes_with_sampling_options() {
        let stochastic = reply_config(
            r#"{"messages":[],"temperature":0.7,"top_p":0.9,"top_k":40,"response_format":{"type":"json_object"}}"#,
        );
        assert_eq!(
            stochastic.grammar,
            Some(Grammar::JsonSchema {
                schema: JSON_OBJECT_SCHEMA.to_string(),
            })
        );
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
        assert_eq!(
            greedy.grammar,
            Some(Grammar::JsonSchema {
                schema: JSON_OBJECT_SCHEMA.to_string(),
            })
        );
        assert_eq!(
            greedy.sampling_policy,
            uzu::types::basic::SamplingPolicy::Custom {
                method: SamplingMethod::Greedy {},
            }
        );
    }
}
