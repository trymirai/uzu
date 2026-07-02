mod error;

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

pub use error::ClassificationSessionError;
use futures::StreamExt;
use hanashi::ClassificationEncoding;
use serde::{Deserialize, Serialize};
use shoji::{
    traits::{Backend, State as StateTrait, backend::classification::Instance},
    types::{
        basic::CancelToken,
        model::{Model, ModelSpecialization},
        session::classification::{
            ClassificationMessage, ClassificationOutput, ClassificationOutputProbabilities, ClassificationStats,
        },
    },
};
use tokio::sync::{Mutex, mpsc};

#[bindings::export(Enumeration)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClassificationSessionState {
    Idle,
    Classifying,
}

struct InstanceHolder {
    instance: Box<dyn Instance>,
    state: Box<dyn StateTrait>,
    encoding: ClassificationEncoding,
}

#[bindings::export(Class)]
#[derive(Clone)]
pub struct ClassificationSession {
    holder: Arc<Mutex<InstanceHolder>>,
    state: Arc<Mutex<ClassificationSessionState>>,
}

impl ClassificationSession {
    pub async fn new(
        backend: Arc<dyn Backend>,
        model: Model,
        path: Option<String>,
    ) -> Result<Self, ClassificationSessionError> {
        if !model.specializations.contains(&ModelSpecialization::Classification {}) {
            return Err(ClassificationSessionError::UnsupportedModel {});
        }
        let reference = path.unwrap_or_else(|| model.identifier.clone());

        let holder = tokio::spawn(async move {
            let classification_backend =
                backend.as_classification_capable().ok_or(ClassificationSessionError::UnsupportedModel {})?;
            let instance = classification_backend.instance(reference.clone(), ()).await.map_err(|error| {
                ClassificationSessionError::Backend {
                    message: error.to_string(),
                }
            })?;
            let instance_state = instance.state().await.map_err(|error| ClassificationSessionError::Backend {
                message: error.to_string(),
            })?;
            let encoding =
                ClassificationEncoding::new(reference.as_str()).map_err(|err| ClassificationSessionError::Backend {
                    message: err.to_string(),
                })?;

            Ok(InstanceHolder {
                instance,
                state: instance_state,
                encoding,
            })
        })
        .await
        .map_err(|error| ClassificationSessionError::Backend {
            message: error.to_string(),
        })??;

        Ok(Self {
            holder: Arc::new(Mutex::new(holder)),
            state: Arc::new(Mutex::new(ClassificationSessionState::Idle)),
        })
    }
}

#[bindings::export(Implementation)]
impl ClassificationSession {
    #[bindings::export(Method(Getter))]
    pub async fn state(&self) -> ClassificationSessionState {
        *self.state.lock().await
    }

    #[bindings::export(Method)]
    pub async fn classify(
        &self,
        input: Vec<ClassificationMessage>,
    ) -> Result<ClassificationOutput, ClassificationSessionError> {
        let time_start = Instant::now();

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

            let mut duration_preprocessing = Duration::from_millis(0);
            let mut duration_forward_pass = Duration::from_millis(0);
            let mut tokens_count = 0usize;
            let result = {
                let mut holder = holder.lock().await;
                let InstanceHolder {
                    instance,
                    state: backend_state,
                    encoding,
                } = &mut *holder;

                let time_before_encoding = Instant::now();
                match encoding.encode(input.as_slice()) {
                    Ok(tokens) => {
                        tokens_count = tokens.len();
                        let input = tokens.iter().map(|token| u64::from(*token)).collect();
                        duration_preprocessing = Instant::now().duration_since(time_before_encoding);

                        let time_before_forward_pass = Instant::now();
                        let mut stream = instance.stream(&input, backend_state.as_mut(), (), cancel_token);
                        let output = stream.next().await;
                        duration_forward_pass = time_before_forward_pass.elapsed();
                        drop(stream);

                        match output {
                            Some(Ok(output)) => Ok(output),
                            Some(Err(error)) => Err(ClassificationSessionError::Backend {
                                message: error.to_string(),
                            }),
                            None => Err(ClassificationSessionError::NoResponse {}),
                        }
                    },
                    Err(err) => Err(ClassificationSessionError::Backend {
                        message: err.to_string(),
                    }),
                }
            };

            let output = result.map(|result| {
                let time_before_postprocessing = Instant::now();
                let logits_f64 = result.logits.iter().map(|&value| value as f64).collect::<Vec<f64>>();
                let (predicted_label, confidence) = result
                    .probabilities
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(label, prob)| (label.clone(), *prob as f64))
                    .unwrap_or((String::new(), 0.0f64));

                let preprocessing_duration = duration_preprocessing.as_secs_f64();
                let forward_pass_duration = duration_forward_pass.as_secs_f64();

                let postprocessing_duration = time_before_postprocessing.elapsed().as_secs_f64();
                let total_duration = time_start.elapsed().as_secs_f64();

                let tokens_per_second = if forward_pass_duration > 0.0 {
                    tokens_count as f64 / forward_pass_duration
                } else {
                    0.0f64
                };

                ClassificationOutput {
                    logits: logits_f64,
                    probabilities: ClassificationOutputProbabilities {
                        values: result.probabilities.into_iter().map(|entry| (entry.0, entry.1 as f64)).collect(),
                    },
                    stats: ClassificationStats {
                        preprocessing_duration,
                        forward_pass_duration,
                        postprocessing_duration,
                        total_duration,
                        tokens_count: tokens_count as i64,
                        tokens_per_second,
                        predicted_label,
                        confidence,
                    },
                }
            });
            let _ = sender.send(output);

            *state.lock().await = ClassificationSessionState::Idle;
        });

        receiver.recv().await.unwrap_or(Err(ClassificationSessionError::NoResponse {}))
    }
}
