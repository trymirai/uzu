mod error;
pub mod message;
pub mod token;

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

pub use error::Error;
use futures::{Stream as FuturesStream, StreamExt};
use indexmap::IndexMap;
use shoji::{
    traits::{
        Backend,
        backend::chat_message::{Output as BackendOutput, ToolCallState},
    },
    types::{
        basic::{CancellationToken, Value},
        encoding::{Message, ToolCall},
        model::{Model, Specialization},
        session::chat::{Config, Output, StreamConfig},
    },
};
use tokio::sync::{Mutex, mpsc};

pub struct Stream {
    receiver: mpsc::UnboundedReceiver<Result<Vec<Output>, Error>>,
}

impl Stream {
    pub async fn next(&mut self) -> Option<Result<Vec<Output>, Error>> {
        self.receiver.recv().await
    }
}

impl FuturesStream for Stream {
    type Item = Result<Vec<Output>, Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(context)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum State {
    Idle,
    Generation,
    ToolCalling,
    Resetting,
}

enum Instance {
    Token(token::Session),
    Message(message::Session),
}

pub struct Session {
    instance: Arc<Mutex<Instance>>,
    state: Arc<Mutex<State>>,
    messages: Arc<Mutex<Vec<Message>>>,
}

impl Session {
    pub async fn new(
        backend: &dyn Backend,
        config: Config,
        model: Model,
        path: Option<String>,
    ) -> Result<Self, Error> {
        if !model.specializations.contains(&Specialization::Chat) {
            return Err(Error::UnsupportedModel);
        }
        let reference = path.unwrap_or_else(|| model.identifier.clone());

        let instance = if let Some(token_backend) = backend.as_chat_via_token_capable() {
            Instance::Token(token::Session::new(token_backend, config, reference).await?)
        } else if let Some(message_backend) = backend.as_chat_via_message_capable() {
            Instance::Message(message::Session::new(message_backend, config, reference).await?)
        } else {
            return Err(Error::UnsupportedModel);
        };

        Ok(Self {
            instance: Arc::new(Mutex::new(instance)),
            state: Arc::new(Mutex::new(State::Idle)),
            messages: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub async fn state(&self) -> State {
        *self.state.lock().await
    }

    pub async fn messages(&self) -> Vec<Message> {
        self.messages.lock().await.clone()
    }

    pub async fn reset(&self) -> Result<(), Error> {
        {
            let mut state = self.state.lock().await;
            match *state {
                State::Idle | State::ToolCalling => {
                    *state = State::Resetting;
                },
                State::Generation | State::Resetting => {
                    return Err(Error::UnableToPerformOperationInCurrentState);
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
        *self.state.lock().await = State::Idle;

        result
    }

    pub async fn run(
        &self,
        input: Vec<Message>,
        config: StreamConfig,
    ) -> Result<Vec<Output>, Error> {
        let (mut stream, _) = self.stream(input, config);
        let mut outputs: Option<Vec<Output>> = None;
        while let Some(progress) = stream.next().await {
            match progress {
                Ok(current_outputs) => outputs = Some(current_outputs.clone()),
                Err(error) => return Err(error),
            }
        }
        outputs.ok_or(Error::NoResponse)
    }

    pub fn stream(
        &self,
        input: Vec<Message>,
        config: StreamConfig,
    ) -> (Stream, CancellationToken) {
        let cancel_token_to_return = CancellationToken::new();
        let (sender, receiver) = mpsc::unbounded_channel::<Result<Vec<Output>, Error>>();

        let instance = self.instance.clone();
        let state = self.state.clone();
        let messages = self.messages.clone();
        let cancel_token = cancel_token_to_return.inner().clone();

        tokio::spawn(async move {
            {
                let mut state = state.lock().await;
                match *state {
                    State::Idle => {
                        *state = State::Generation;
                    },
                    State::Generation | State::Resetting | State::ToolCalling => {
                        let _ = sender.send(Err(Error::UnableToPerformOperationInCurrentState));
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

            let mut outputs: IndexMap<u32, Output> = IndexMap::new();

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
                        let output = Output {
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
                *state = State::Idle;
            }
        });

        (
            Stream {
                receiver,
            },
            cancel_token_to_return,
        )
    }
}

fn build_message(output: &BackendOutput) -> Message {
    let mut message = Message::assistant();
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
