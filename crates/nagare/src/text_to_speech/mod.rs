mod error;

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

pub use error::TextToSpeechSessionError;
use futures::{Stream as FuturesStream, StreamExt};
use serde::{Deserialize, Serialize};
use shoji::{
    traits::{Backend, State as StateTrait, backend::text_to_speech::Instance},
    types::{
        basic::{CancellationToken, PcmBatch},
        model::{Model, ModelSpecialization},
    },
};
use tokio::sync::{Mutex, mpsc};

#[bindings::export(Class)]
pub struct TextToSpeechSessionStream {
    receiver: Mutex<mpsc::UnboundedReceiver<Result<PcmBatch, TextToSpeechSessionError>>>,
}

impl TextToSpeechSessionStream {
    pub async fn next(&self) -> Option<Result<PcmBatch, TextToSpeechSessionError>> {
        self.receiver.lock().await.recv().await
    }
}

impl FuturesStream for TextToSpeechSessionStream {
    type Item = Result<PcmBatch, TextToSpeechSessionError>;

    fn poll_next(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.get_mut().receiver.get_mut().poll_recv(context)
    }
}

#[bindings::export(Enum)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TextToSpeechSessionState {
    Idle,
    Synthesizing,
}

struct InstanceHolder {
    instance: Box<dyn Instance>,
    state: Box<dyn StateTrait>,
}

#[bindings::export(Class)]
pub struct TextToSpeechSession {
    holder: Arc<Mutex<InstanceHolder>>,
    state: Arc<Mutex<TextToSpeechSessionState>>,
}

impl TextToSpeechSession {
    pub async fn new(
        backend: &dyn Backend,
        model: Model,
        path: Option<String>,
    ) -> Result<Self, TextToSpeechSessionError> {
        if !model.specializations.contains(&ModelSpecialization::TextToSpeech) {
            return Err(TextToSpeechSessionError::UnsupportedModel);
        }
        let reference = path.unwrap_or_else(|| model.identifier.clone());
        let text_to_speech_backend =
            backend.as_text_to_speech_capable().ok_or(TextToSpeechSessionError::UnsupportedModel)?;
        let instance = text_to_speech_backend.instance(reference, ()).await.map_err(|error| {
            TextToSpeechSessionError::Backend {
                message: error.to_string(),
            }
        })?;
        let instance_state = instance.state().await.map_err(|error| TextToSpeechSessionError::Backend {
            message: error.to_string(),
        })?;
        Ok(Self {
            holder: Arc::new(Mutex::new(InstanceHolder {
                instance,
                state: instance_state,
            })),
            state: Arc::new(Mutex::new(TextToSpeechSessionState::Idle)),
        })
    }
}

#[bindings::export(Implementation)]
impl TextToSpeechSession {
    #[bindings::export(Method)]
    pub async fn state(&self) -> TextToSpeechSessionState {
        *self.state.lock().await
    }

    #[bindings::export(Method)]
    pub async fn synthesize(
        &self,
        input: String,
    ) -> Result<PcmBatch, TextToSpeechSessionError> {
        let (stream, _) = self.synthesize_stream(input);
        let mut batches: Vec<PcmBatch> = Vec::new();
        while let Some(event) = stream.next().await {
            match event {
                Ok(batch) => batches.push(batch),
                Err(error) => return Err(error),
            }
        }
        if let Some(batch) = batches.last() {
            return Ok(batch.clone());
        }
        Err(TextToSpeechSessionError::NoResponse)
    }
}

impl TextToSpeechSession {
    fn synthesize_stream(
        &self,
        input: String,
    ) -> (TextToSpeechSessionStream, CancellationToken) {
        let cancel_token_to_return = CancellationToken::new();
        let (sender, receiver) = mpsc::unbounded_channel::<Result<PcmBatch, TextToSpeechSessionError>>();

        let holder = self.holder.clone();
        let state = self.state.clone();
        let cancel_token = cancel_token_to_return.inner().clone();

        tokio::spawn(async move {
            {
                let mut state = state.lock().await;
                match *state {
                    TextToSpeechSessionState::Idle => {
                        *state = TextToSpeechSessionState::Synthesizing;
                    },
                    TextToSpeechSessionState::Synthesizing => {
                        let _ = sender.send(Err(TextToSpeechSessionError::UnableToPerformOperationInCurrentState));
                        return;
                    },
                }
            }

            {
                let mut holder = holder.lock().await;
                let InstanceHolder {
                    instance,
                    state: backend_state,
                } = &mut *holder;

                let mut stream = instance.stream(&input, backend_state.as_mut(), (), cancel_token);
                while let Some(event) = stream.next().await {
                    let item = event.map_err(|error| TextToSpeechSessionError::Backend {
                        message: error.to_string(),
                    });
                    if sender.send(item).is_err() {
                        break;
                    }
                }
            }

            *state.lock().await = TextToSpeechSessionState::Idle;
        });

        (
            TextToSpeechSessionStream {
                receiver: Mutex::new(receiver),
            },
            cancel_token_to_return,
        )
    }
}
