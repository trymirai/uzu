mod error;

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

pub use error::Error;
use futures::{Stream as FuturesStream, StreamExt};
use shoji::{
    traits::{Backend, State as StateTrait, backend::text_to_speech::Instance},
    types::{
        basic::CancellationToken,
        model::{Model, ModelSpecialization},
        session::text_to_speech::PcmBatch,
    },
};
use tokio::sync::{Mutex, mpsc};

pub struct Stream {
    receiver: mpsc::UnboundedReceiver<Result<PcmBatch, Error>>,
}

impl Stream {
    pub async fn next(&mut self) -> Option<Result<PcmBatch, Error>> {
        self.receiver.recv().await
    }
}

impl FuturesStream for Stream {
    type Item = Result<PcmBatch, Error>;

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
    Synthesizing,
}

struct InstanceHolder {
    instance: Box<dyn Instance>,
    state: Box<dyn StateTrait>,
}

pub struct Session {
    holder: Arc<Mutex<InstanceHolder>>,
    state: Arc<Mutex<State>>,
}

impl Session {
    pub async fn new(
        backend: &dyn Backend,
        model: Model,
        path: Option<String>,
    ) -> Result<Self, Error> {
        if !model.specializations.contains(&ModelSpecialization::TextToSpeech) {
            return Err(Error::UnsupportedModel);
        }
        let reference = path.unwrap_or_else(|| model.identifier.clone());
        let text_to_speech_backend = backend.as_text_to_speech_capable().ok_or(Error::UnsupportedModel)?;
        let instance = text_to_speech_backend.instance(reference, ()).await.map_err(|error| Error::Backend {
            message: error.to_string(),
        })?;
        let instance_state = instance.state().await.map_err(|error| Error::Backend {
            message: error.to_string(),
        })?;
        Ok(Self {
            holder: Arc::new(Mutex::new(InstanceHolder {
                instance,
                state: instance_state,
            })),
            state: Arc::new(Mutex::new(State::Idle)),
        })
    }

    pub async fn state(&self) -> State {
        *self.state.lock().await
    }

    pub async fn synthesize(
        &self,
        input: String,
    ) -> Result<PcmBatch, Error> {
        let (mut stream, _) = self.synthesize_stream(input);
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
        Err(Error::NoResponse)
    }

    fn synthesize_stream(
        &self,
        input: String,
    ) -> (Stream, CancellationToken) {
        let cancel_token_to_return = CancellationToken::new();
        let (sender, receiver) = mpsc::unbounded_channel::<Result<PcmBatch, Error>>();

        let holder = self.holder.clone();
        let state = self.state.clone();
        let cancel_token = cancel_token_to_return.inner().clone();

        tokio::spawn(async move {
            {
                let mut state = state.lock().await;
                match *state {
                    State::Idle => {
                        *state = State::Synthesizing;
                    },
                    State::Synthesizing => {
                        let _ = sender.send(Err(Error::UnableToPerformOperationInCurrentState));
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
                    let item = event.map_err(|error| Error::Backend {
                        message: error.to_string(),
                    });
                    if sender.send(item).is_err() {
                        break;
                    }
                }
            }

            *state.lock().await = State::Idle;
        });

        (
            Stream {
                receiver,
            },
            cancel_token_to_return,
        )
    }
}
