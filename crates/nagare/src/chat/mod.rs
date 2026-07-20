mod error;
pub mod message;
pub mod token;

use std::sync::Arc;

pub use error::ChatSessionError;
use futures::StreamExt;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use shoji::{
    traits::{
        Backend,
        backend::chat_message::{Output as BackendOutput, ToolCallState},
    },
    types::{
        basic::{CancelToken, ToolCall, Value},
        model::{Model, ModelSpecialization},
        session::chat::{
            ChatConfig, ChatContentBlock, ChatMessage, ChatReply, ChatReplyConfig, ChatReplyFinishReason, ChatRole,
        },
    },
};
use tokio::sync::{Mutex, mpsc, mpsc::UnboundedSender};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    telemetry::{Telemetry, TelemetryEvent},
    tool::{func_def::ToolFunctionDefinition, registry::ToolRegistry},
};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChatSessionStreamChunk {
    Replies {
        replies: Vec<ChatReply>,
    },
    Error {
        error: ChatSessionError,
    },
}

#[bindings::export(Class(Stream))]
#[derive(Clone)]
pub struct ChatSessionStream {
    receiver: Arc<Mutex<mpsc::UnboundedReceiver<Result<Vec<ChatReply>, ChatSessionError>>>>,
    cancel_token: CancelToken,
}

#[bindings::export(Implementation)]
impl ChatSessionStream {
    #[bindings::export(Method(StreamNext))]
    pub async fn next(&self) -> Option<ChatSessionStreamChunk> {
        match self.receiver.lock().await.recv().await {
            Some(Ok(replies)) => Some(ChatSessionStreamChunk::Replies {
                replies,
            }),
            Some(Err(error)) => Some(ChatSessionStreamChunk::Error {
                error,
            }),
            None => None,
        }
    }

    #[bindings::export(Method(Getter))]
    pub fn cancel_token(&self) -> CancelToken {
        self.cancel_token.clone()
    }
}

#[bindings::export(Enumeration)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChatSessionState {
    Idle,
    Generation,
    ToolCalling,
    Resetting,
}

enum Instance {
    Token(token::Session),
    Message(message::Session),
}

impl Instance {
    pub fn peak_memory_usage(&self) -> Option<usize> {
        match self {
            Instance::Token(session) => session.peak_memory_usage(),
            Instance::Message(session) => session.peak_memory_usage(),
        }
    }

    pub fn supports_tool_calls(&self) -> bool {
        match self {
            Instance::Token(session) => session.supports_tool_calls(),
            Instance::Message(session) => session.supports_tool_calls(),
        }
    }

    pub fn supports_multiple_tool_calls(&self) -> bool {
        match self {
            Instance::Token(session) => session.supports_multiple_tool_calls(),
            Instance::Message(session) => session.supports_multiple_tool_calls(),
        }
    }
}

#[bindings::export(Class)]
#[derive(Clone)]
pub struct ChatSession {
    instance: Arc<Mutex<Instance>>,
    state: Arc<Mutex<ChatSessionState>>,
    messages: Arc<Mutex<Vec<ChatMessage>>>,
    model_id: String,
    telemetry: Telemetry,
    tool_registry: Option<Arc<Mutex<ToolRegistry>>>,
}

impl ChatSession {
    pub async fn new(
        backend: Arc<dyn Backend>,
        config: ChatConfig,
        model: Model,
        path: Option<String>,
        telemetry: Telemetry,
    ) -> Result<Self, ChatSessionError> {
        if !model.specializations.contains(&ModelSpecialization::Chat {}) {
            return Err(ChatSessionError::UnsupportedModel {});
        }
        let model_id = model.identifier.clone();
        let reference = path.unwrap_or_else(|| model.identifier.clone());

        let instance = tokio::spawn(async move {
            if let Some(token_backend) = backend.as_chat_via_token_capable() {
                token::Session::new(token_backend, config, reference, &model).await.map(Instance::Token)
            } else if let Some(message_backend) = backend.as_chat_via_message_capable() {
                message::Session::new(message_backend, config, reference).await.map(Instance::Message)
            } else {
                Err(ChatSessionError::UnsupportedModel {})
            }
        })
        .await
        .map_err(|error| ChatSessionError::Backend {
            message: error.to_string(),
        })??;

        let supports_tool_calls = instance.supports_tool_calls();
        Ok(Self {
            instance: Arc::new(Mutex::new(instance)),
            state: Arc::new(Mutex::new(ChatSessionState::Idle)),
            messages: Arc::new(Mutex::new(Vec::new())),
            model_id,
            telemetry,
            tool_registry: supports_tool_calls.then(|| Arc::new(Mutex::new(ToolRegistry::new()))),
        })
    }

    pub async fn peak_memory_usage(&self) -> Option<usize> {
        self.instance.lock().await.peak_memory_usage()
    }

    pub async fn add_tool_functions(
        &mut self,
        definitions: Vec<ToolFunctionDefinition>,
    ) -> Result<(), ChatSessionError> {
        if definitions.is_empty() {
            return Ok(());
        }
        if self.tool_registry.is_none() {
            return Err(ChatSessionError::Backend {
                message: "Tool calls are not supported by this model".to_string(),
            });
        }

        let mut messages_guard = self.messages.lock().await;
        let empty_messages = messages_guard.is_empty();
        if !empty_messages {
            messages_guard.retain(|msg| !contains_tools_definitions(msg));
        }
        drop(messages_guard);

        if !empty_messages {
            let mut instance_guard = self.instance.lock().await;
            match &mut *instance_guard {
                Instance::Token(session) => session.reset().await?,
                Instance::Message(session) => session.reset().await?,
            };
        }

        if let Some(registry) = self.tool_registry.as_mut() {
            let mut registry_guard = registry.lock().await;
            for def in definitions {
                registry_guard.add_function(def)
            }
        }

        Ok(())
    }

    async fn send_input(
        &self,
        sender: UnboundedSender<Result<Vec<ChatReply>, ChatSessionError>>,
        input: Vec<ChatMessage>,
        config: ChatReplyConfig,
        cancel_token: CancellationToken,
    ) {
        // prepare all messages
        let all_messages = {
            let mut messages_guard = self.messages.lock().await;
            messages_guard.extend(input);

            // register tools if needed
            if let Some(ref registry) = self.tool_registry {
                let namespaces = registry.lock().await.get_namespaces();
                if !namespaces.is_empty() && !messages_guard.iter().any(contains_tools_definitions) {
                    let position = messages_guard.iter().position(|msg| msg.role == ChatRole::System {});
                    let tools_msg = ChatMessage::developer().with_tool_namespaces(namespaces);
                    messages_guard.insert(position.map(|pos| pos + 1).unwrap_or(0), tools_msg);
                }
            }

            messages_guard.clone()
        };

        self.telemetry.report(TelemetryEvent::ModelInferenceStarted {
            model_id: self.model_id.clone(),
        });

        let mut outputs: IndexMap<u32, ChatReply> = IndexMap::new();
        let mut error_value: Option<serde_json::Value> = None;
        let mut generated_tool_call_identifiers: Vec<Option<String>> = Vec::new();

        let mut instance = self.instance.lock().await;
        let mut stream = match &mut *instance {
            Instance::Token(session) => session.stream(&all_messages, config, cancel_token).await,
            Instance::Message(session) => session.stream(&all_messages, config, cancel_token),
        };
        while let Some(partial_output) = stream.next().await {
            match partial_output {
                Ok(backend_output) => {
                    // prepare reply
                    let message = build_message(&backend_output, &mut generated_tool_call_identifiers);
                    let finish_reason = backend_output.finish_reason;
                    let output = ChatReply {
                        message: message.clone(),
                        stats: backend_output.stats.clone(),
                        finish_reason: finish_reason.clone(),
                    };

                    // add output to messages list
                    let mut messages_guard = self.messages.lock().await;
                    let is_new = outputs.insert(0, output).is_none();
                    if is_new {
                        messages_guard.push(message.clone());
                    } else if let Some(last) = messages_guard.last_mut() {
                        *last = message.clone();
                    }
                    drop(messages_guard);

                    // send new output
                    if sender.send(Ok(outputs.values().cloned().collect())).is_err() {
                        break;
                    }
                    if finish_reason.is_some() {
                        break;
                    }
                },
                Err(error) => {
                    error_value = Some(serde_json::json!({ "message": error.to_string() }));
                    let _ = sender.send(Err(error));
                    break;
                },
            }
        }
        drop(stream);
        drop(instance);

        // telemetry report result
        if let Some(error) = error_value {
            self.telemetry.report(TelemetryEvent::ModelInferenceFailed {
                error,
            });
        } else if let Some(last) = outputs.values().last() {
            self.telemetry.report(TelemetryEvent::ModelInferenceFinished {
                model_id: self.model_id.clone(),
                stats: last.stats.clone(),
            });
        }
    }

    async fn execute_turn(
        &self,
        sender: UnboundedSender<Result<Vec<ChatReply>, ChatSessionError>>,
        input: Vec<ChatMessage>,
        config: ChatReplyConfig,
        cancel_token: CancellationToken,
    ) {
        // check state
        if !self.try_transition(ChatSessionState::Idle, ChatSessionState::Generation).await {
            let _ = sender.send(Err(ChatSessionError::UnableToPerformOperationInCurrentState {}));
            return;
        }

        let mut next_input = input;

        'turns: loop {
            let (turn_sender, mut turn_receiver) =
                mpsc::unbounded_channel::<Result<Vec<ChatReply>, ChatSessionError>>();
            let turn_input = std::mem::take(&mut next_input);
            let turn_config = config.clone();
            let turn_cancel_token = cancel_token.clone();
            let session = self.clone();
            tokio::spawn(
                async move { session.send_input(turn_sender, turn_input, turn_config, turn_cancel_token).await },
            );

            // send message
            let mut turn_replies: Vec<ChatReply> = Vec::new();
            while let Some(result) = turn_receiver.recv().await {
                match result {
                    Ok(replies) => {
                        turn_replies = replies;
                        if sender.send(Ok(turn_replies.clone())).is_err() {
                            break 'turns;
                        }
                    },
                    Err(error) => {
                        let _ = sender.send(Err(error));
                        break 'turns;
                    },
                }
            }

            // check for tool calls and execute if needed
            if let Some(last_reply) = turn_replies.last()
                && last_reply.finish_reason == Some(ChatReplyFinishReason::ToolCalls)
                && !cancel_token.is_cancelled()
            {
                if self.try_transition(ChatSessionState::Generation, ChatSessionState::ToolCalling).await {
                    let mut tool_calls = last_reply.message.tool_calls();
                    let supports_multiple_tool_calls = self.instance.lock().await.supports_multiple_tool_calls();
                    if !supports_multiple_tool_calls && tool_calls.len() > 1 {
                        tool_calls.drain(1..);
                        // keep the history renderable: templates with a single-call limit
                        // cannot re-encode an assistant message holding extra calls
                        let mut messages_guard = self.messages.lock().await;
                        if let Some(last_message) = messages_guard.last_mut() {
                            let mut tool_call_seen = false;
                            last_message.content.retain(|block| match block {
                                ChatContentBlock::ToolCall {
                                    ..
                                }
                                | ChatContentBlock::ToolCallCandidate {
                                    ..
                                } => !std::mem::replace(&mut tool_call_seen, true),
                                _ => true,
                            });
                        }
                    }

                    let tool_messages = tokio::select! {
                        tool_messages = self.execute_tool_calls(tool_calls) => tool_messages,
                        _ = cancel_token.cancelled() => break,
                    };
                    if self.try_transition(ChatSessionState::ToolCalling, ChatSessionState::Generation).await {
                        if !tool_messages.is_empty() && !cancel_token.is_cancelled() {
                            next_input = tool_messages;
                        } else {
                            break;
                        }
                    } else {
                        return;
                    }
                } else {
                    return;
                }
            } else {
                break;
            }
        }

        *self.state.lock().await = ChatSessionState::Idle;
    }

    async fn execute_tool_calls(
        &self,
        tool_calls: Vec<ToolCall>,
    ) -> Vec<ChatMessage> {
        if tool_calls.is_empty() {
            return vec![];
        }
        let Some(ref registry) = self.tool_registry else {
            return vec![];
        };

        let registry_guard = registry.lock().await;
        let mut tool_messages: Vec<ChatMessage> = Vec::with_capacity(tool_calls.len());
        for call in tool_calls {
            let function: Option<&ToolFunctionDefinition> = (*registry_guard).get_function(&call.name);
            let value = if let Some(func) = function {
                func.execute(call.arguments).await.unwrap_or_else(|err| {
                    Value::from(serde_json::json!({
                        "error": err.to_string(),
                    }))
                })
            } else {
                Value::from(serde_json::json!({
                    "error": format!("Unknown function: {}", call.name),
                }))
            };

            // chat templates expect tool results as plain text, one message per call
            let tool_message = ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
                identifier: call.identifier.clone(),
                name: Some(call.name.clone()),
                value,
            });
            tool_messages.push(tool_message);
        }
        tool_messages
    }

    /// Atomically switches state from `from` to `to`.
    /// Returns false (leaving state untouched) if the current state differs.
    async fn try_transition(
        &self,
        from: ChatSessionState,
        to: ChatSessionState,
    ) -> bool {
        let mut state_guard = self.state.lock().await;
        if *state_guard == from {
            *state_guard = to;
            true
        } else {
            false
        }
    }
}

#[bindings::export(Implementation)]
impl ChatSession {
    #[bindings::export(Method(Getter))]
    pub async fn state(&self) -> ChatSessionState {
        *self.state.lock().await
    }

    #[bindings::export(Method(Getter))]
    pub async fn messages(&self) -> Vec<ChatMessage> {
        self.messages.lock().await.clone()
    }

    #[bindings::export(Method)]
    pub async fn reset(&self) -> Result<(), ChatSessionError> {
        {
            let mut state = self.state.lock().await;
            match *state {
                ChatSessionState::Idle | ChatSessionState::ToolCalling => {
                    *state = ChatSessionState::Resetting;
                },
                ChatSessionState::Generation | ChatSessionState::Resetting => {
                    return Err(ChatSessionError::UnableToPerformOperationInCurrentState {});
                },
            }
        }

        let result = {
            let mut guard = self.instance.lock().await;
            match &mut *guard {
                Instance::Token(session) => session.reset().await,
                Instance::Message(session) => session.reset().await,
            }
        };

        self.messages.lock().await.clear();
        *self.state.lock().await = ChatSessionState::Idle;

        result
    }

    #[bindings::export(Method)]
    pub async fn reply(
        &self,
        input: Vec<ChatMessage>,
        config: ChatReplyConfig,
    ) -> Result<Vec<ChatReply>, ChatSessionError> {
        let stream = self.reply_with_stream(input, config).await;
        let mut outputs: Option<Vec<ChatReply>> = None;
        while let Some(progress) = stream.next().await {
            match progress {
                ChatSessionStreamChunk::Replies {
                    replies,
                } => outputs = Some(replies.clone()),
                ChatSessionStreamChunk::Error {
                    error,
                } => return Err(error),
            }
        }
        outputs.ok_or(ChatSessionError::NoResponse {})
    }

    #[bindings::export(Method)]
    pub async fn reply_with_stream(
        &self,
        input: Vec<ChatMessage>,
        config: ChatReplyConfig,
    ) -> ChatSessionStream {
        let cancel_token_to_return = CancelToken::new();
        let cancel_token = cancel_token_to_return.inner().clone();
        let (sender, receiver) = mpsc::unbounded_channel::<Result<Vec<ChatReply>, ChatSessionError>>();
        let session = self.clone();
        tokio::spawn(async move { session.execute_turn(sender, input, config, cancel_token).await });
        ChatSessionStream {
            receiver: Arc::new(Mutex::new(receiver)),
            cancel_token: cancel_token_to_return,
        }
    }
}

fn build_message(
    output: &BackendOutput,
    generated_tool_call_identifiers: &mut Vec<Option<String>>,
) -> ChatMessage {
    let mut message = ChatMessage::assistant();
    if let Some(reasoning) = &output.reasoning {
        message = message.with_reasoning(reasoning.clone());
    }
    if let Some(text) = &output.text {
        message = message.with_text(text.clone());
    }
    for (index, tool_call) in output.tool_calls.iter().enumerate() {
        message = match tool_call {
            ToolCallState::Candidate(candidate) => {
                message.with_tool_call_candidate(Value::from(serde_json::Value::String(candidate.clone())))
            },
            ToolCallState::Finished(tool_call) => {
                let identifier = tool_call.identifier.clone().or_else(|| {
                    if generated_tool_call_identifiers.len() <= index {
                        generated_tool_call_identifiers.resize(index + 1, None);
                    }
                    Some(
                        generated_tool_call_identifiers[index]
                            .get_or_insert_with(|| Uuid::new_v4().to_string())
                            .clone(),
                    )
                });
                message.with_tool_call(ToolCall {
                    identifier,
                    name: tool_call.name.clone(),
                    arguments: tool_call.arguments.clone(),
                })
            },
        };
    }
    message
}

fn contains_tools_definitions(msg: &ChatMessage) -> bool {
    msg.role == ChatRole::Developer {}
        && msg.content.iter().any(|content| matches!(content, ChatContentBlock::Tools { .. }))
}
