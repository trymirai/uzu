mod error;

use std::sync::Arc;

pub use error::ClassificationSessionError;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use shoji::{
    traits::{Backend, State as StateTrait, backend::classification::Instance},
    types::{
        basic::CancelToken,
        model::{Model, ModelSpecialization},
        session::classification::{ClassificationMessage, ClassificationOutput},
    },
};
use tokio::sync::{Mutex, mpsc};

#[bindings::export(Enum)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClassificationSessionState {
    Idle,
    Classifying,
}

struct InstanceHolder {
    instance: Box<dyn Instance>,
    state: Box<dyn StateTrait>,
}

#[bindings::export(Class)]
pub struct ClassificationSession {
    holder: Arc<Mutex<InstanceHolder>>,
    state: Arc<Mutex<ClassificationSessionState>>,
}

impl ClassificationSession {
    pub async fn new(
        backend: &dyn Backend,
        model: Model,
        path: Option<String>,
    ) -> Result<Self, ClassificationSessionError> {
        if !model.specializations.contains(&ModelSpecialization::Classification {}) {
            return Err(ClassificationSessionError::UnsupportedModel {});
        }
        let reference = path.unwrap_or_else(|| model.identifier.clone());
        let classification_backend =
            backend.as_classification_capable().ok_or(ClassificationSessionError::UnsupportedModel {})?;
        let instance = classification_backend.instance(reference, ()).await.map_err(|error| {
            ClassificationSessionError::Backend {
                message: error.to_string(),
            }
        })?;
        let instance_state = instance.state().await.map_err(|error| ClassificationSessionError::Backend {
            message: error.to_string(),
        })?;
        Ok(Self {
            holder: Arc::new(Mutex::new(InstanceHolder {
                instance,
                state: instance_state,
            })),
            state: Arc::new(Mutex::new(ClassificationSessionState::Idle)),
        })
    }
}

#[bindings::export(Implementation)]
impl ClassificationSession {
    #[bindings::export(Getter)]
    pub async fn state(&self) -> ClassificationSessionState {
        *self.state.lock().await
    }

    #[bindings::export(Method)]
    pub async fn classify(
        &self,
        input: Vec<ClassificationMessage>,
    ) -> Result<ClassificationOutput, ClassificationSessionError> {
        let cancel_token_to_return = CancelToken::new();
        let (sender, mut receiver) =
            mpsc::unbounded_channel::<Result<ClassificationOutput, ClassificationSessionError>>();

        let holder = self.holder.clone();
        let state = self.state.clone();
        let cancel_token = cancel_token_to_return.inner().clone();

        tokio::spawn(async move {
            {
                let mut state = state.lock().await;
                match *state {
                    ClassificationSessionState::Idle => {
                        *state = ClassificationSessionState::Classifying;
                    },
                    ClassificationSessionState::Classifying => {
                        let _ = sender.send(Err(ClassificationSessionError::UnableToPerformOperationInCurrentState {}));
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
                    Some(Err(error)) => Err(ClassificationSessionError::Backend {
                        message: error.to_string(),
                    }),
                    None => Err(ClassificationSessionError::NoResponse {}),
                }
            };
            let _ = sender.send(result);

            *state.lock().await = ClassificationSessionState::Idle;
        });

        receiver.recv().await.unwrap_or(Err(ClassificationSessionError::NoResponse {}))
    }
}
