mod error;

use std::sync::Arc;

pub use error::Error;
use futures::StreamExt;
use shoji::{
    traits::{Backend, State as StateTrait, backend::classification::Instance},
    types::{
        basic::CancellationToken,
        model::{Model, ModelSpecialization},
        session::classification::{ClassificationMessage, ClassificationOutput},
    },
};
use tokio::sync::{Mutex, mpsc};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum State {
    Idle,
    Classifying,
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
        if !model.specializations.contains(&ModelSpecialization::Classification) {
            return Err(Error::UnsupportedModel);
        }
        let reference = path.unwrap_or_else(|| model.identifier.clone());
        let classification_backend = backend.as_classification_capable().ok_or(Error::UnsupportedModel)?;
        let instance = classification_backend.instance(reference, ()).await.map_err(|error| Error::Backend {
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

    pub async fn classify(
        &self,
        input: Vec<ClassificationMessage>,
    ) -> Result<ClassificationOutput, Error> {
        let cancel_token_to_return = CancellationToken::new();
        let (sender, mut receiver) = mpsc::unbounded_channel::<Result<ClassificationOutput, Error>>();

        let holder = self.holder.clone();
        let state = self.state.clone();
        let cancel_token = cancel_token_to_return.inner().clone();

        tokio::spawn(async move {
            {
                let mut state = state.lock().await;
                match *state {
                    State::Idle => {
                        *state = State::Classifying;
                    },
                    State::Classifying => {
                        let _ = sender.send(Err(Error::UnableToPerformOperationInCurrentState));
                        return;
                    },
                }
            }

            let result = {
                let mut holder = holder.lock().await;
                let InstanceHolder {
                    instance,
                    state: backend_state,
                } = &mut *holder;

                let mut stream = instance.stream(&input, backend_state.as_mut(), (), cancel_token);
                let output = stream.next().await;
                drop(stream);

                match output {
                    Some(Ok(output)) => Ok(output),
                    Some(Err(error)) => Err(Error::Backend {
                        message: error.to_string(),
                    }),
                    None => Err(Error::NoResponse),
                }
            };
            let _ = sender.send(result);

            *state.lock().await = State::Idle;
        });

        receiver.recv().await.unwrap_or(Err(Error::NoResponse))
    }
}
