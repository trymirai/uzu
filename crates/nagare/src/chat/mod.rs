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
        session::chat::{ChatConfig, ChatMessage, ChatReply, ChatReplyConfig},
    },
};
use tokio::sync::{Mutex, mpsc};

#[bindings::export(Enum)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatSessionStreamChunk {
    Replies {
        replies: Vec<ChatReply>,
    },
    Error {
        error: ChatSessionError,
    },
}

#[bindings::export(Stream)]
#[derive(Clone)]
pub struct ChatSessionStream {
    receiver: Arc<Mutex<mpsc::UnboundedReceiver<Result<Vec<ChatReply>, ChatSessionError>>>>,
    cancel_token: CancelToken,
}

#[bindings::export(Implementation)]
impl ChatSessionStream {
    #[bindings::export(StreamNext)]
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

    #[bindings::export(Getter)]
    pub fn cancel_token(&self) -> CancelToken {
        self.cancel_token.clone()
    }
}

#[bindings::export(Enum)]
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

#[bindings::export(Class)]
pub struct ChatSession {
    instance: Arc<Mutex<Instance>>,
    state: Arc<Mutex<ChatSessionState>>,
    messages: Arc<Mutex<Vec<ChatMessage>>>,
}

impl ChatSession {
    pub async fn new(
        backend: &dyn Backend,
        config: ChatConfig,
        model: Model,
        path: Option<String>,
    ) -> Result<Self, ChatSessionError> {
        if !model.specializations.contains(&ModelSpecialization::Chat {}) {
            return Err(ChatSessionError::UnsupportedModel {});
        }
        let reference = path.unwrap_or_else(|| model.identifier.clone());

        let instance = if let Some(token_backend) = backend.as_chat_via_token_capable() {
            Instance::Token(token::Session::new(token_backend, config, reference).await?)
        } else if let Some(message_backend) = backend.as_chat_via_message_capable() {
            Instance::Message(message::Session::new(message_backend, config, reference).await?)
        } else {
            return Err(ChatSessionError::UnsupportedModel {});
        };

        Ok(Self {
            instance: Arc::new(Mutex::new(instance)),
            state: Arc::new(Mutex::new(ChatSessionState::Idle)),
            messages: Arc::new(Mutex::new(Vec::new())),
        })
    }
}

#[bindings::export(Implementation)]
impl ChatSession {
    #[bindings::export(Getter)]
    pub async fn state(&self) -> ChatSessionState {
        *self.state.lock().await
    }

    #[bindings::export(Getter)]
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
        let (sender, receiver) = mpsc::unbounded_channel::<Result<Vec<ChatReply>, ChatSessionError>>();

        let instance = self.instance.clone();
        let state = self.state.clone();
        let messages = self.messages.clone();
        let cancel_token = cancel_token_to_return.inner().clone();

        tokio::spawn(async move {
            {
                let mut state = state.lock().await;
                match *state {
                    ChatSessionState::Idle => {
                        *state = ChatSessionState::Generation;
                    },
                    ChatSessionState::Generation | ChatSessionState::Resetting | ChatSessionState::ToolCalling => {
                        let _ = sender.send(Err(ChatSessionError::UnableToPerformOperationInCurrentState {}));
                        return;
                    },
                }
                drop(state);
            }

            let all_messages = {
                let mut messages = messages.lock().await;
                messages.extend(input);
                messages.clone()
            };

            let mut outputs: IndexMap<u32, ChatReply> = IndexMap::new();

            let mut instance = instance.lock().await;
            let mut stream = match &mut *instance {
                Instance::Token(session) => session.stream(&all_messages, config, cancel_token),
                Instance::Message(session) => session.stream(&all_messages, config, cancel_token),
            };

            let turn_index: u32 = 0;
            while let Some(partial_output) = stream.next().await {
                match partial_output {
                    Ok(backend_output) => {
                        let message = build_message(&backend_output);
                        let output = ChatReply {
                            message: message.clone(),
                            stats: backend_output.stats.clone(),
                            finish_reason: backend_output.finish_reason.clone(),
                        };
                        let is_new = outputs.insert(turn_index, output).is_none();

                        {
                            let mut messages = messages.lock().await;
                            if is_new {
                                messages.push(message.clone());
                            } else if let Some(last) = messages.last_mut() {
                                *last = message.clone();
                            }
                        }

                        if sender.send(Ok(outputs.values().cloned().collect())).is_err() {
                            break;
                        }
                    },
                    Err(error) => {
                        let _ = sender.send(Err(error));
                        break;
                    },
                }
            }

            drop(stream);
            drop(instance);

            {
                let mut state = state.lock().await;
                *state = ChatSessionState::Idle;
            }
        });

        ChatSessionStream {
            receiver: Arc::new(Mutex::new(receiver)),
            cancel_token: cancel_token_to_return,
        }
    }
}

fn build_message(output: &BackendOutput) -> ChatMessage {
    let mut message = ChatMessage::assistant();
    if let Some(reasoning) = &output.reasoning {
        message = message.with_reasoning(reasoning.clone());
    }
    if let Some(text) = &output.text {
        message = message.with_text(text.clone());
    }
    for tool_call in &output.tool_calls {
        message = match tool_call {
            ToolCallState::Candidate(candidate) => {
                message.with_tool_call_candidate(Value::from(serde_json::Value::String(candidate.clone())))
            },
            ToolCallState::Finished(tool_call) => message.with_tool_call(ToolCall {
                identifier: tool_call.identifier.clone(),
                name: tool_call.name.clone(),
                arguments: tool_call.arguments.clone(),
            }),
        };
    }
    message
}
