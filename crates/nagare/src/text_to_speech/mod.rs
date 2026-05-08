mod error;

use std::{sync::Arc, time::Instant};

pub use error::TextToSpeechSessionError;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use shoji::{
    traits::{Backend, State as StateTrait, backend::text_to_speech::Instance},
    types::{
        basic::{CancelToken, PcmBatch},
        model::{Model, ModelSpecialization},
        session::text_to_speech::{TextToSpeechOutput, TextToSpeechStats},
    },
};
use tokio::sync::{Mutex, mpsc};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TextToSpeechSessionStreamChunk {
    Output {
        output: TextToSpeechOutput,
    },
    Error {
        error: TextToSpeechSessionError,
    },
}

#[bindings::export(Class(Stream))]
#[derive(Clone)]
pub struct TextToSpeechSessionStream {
    receiver: Arc<Mutex<mpsc::UnboundedReceiver<Result<TextToSpeechOutput, TextToSpeechSessionError>>>>,
    cancel_token: CancelToken,
}

#[bindings::export(Implementation)]
impl TextToSpeechSessionStream {
    #[bindings::export(Method(StreamNext))]
    pub async fn next(&self) -> Option<TextToSpeechSessionStreamChunk> {
        match self.receiver.lock().await.recv().await {
            Some(Ok(output)) => Some(TextToSpeechSessionStreamChunk::Output {
                output,
            }),
            Some(Err(error)) => Some(TextToSpeechSessionStreamChunk::Error {
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
pub enum TextToSpeechSessionState {
    Idle,
    Synthesizing,
}

struct InstanceHolder {
    instance: Box<dyn Instance>,
    state: Box<dyn StateTrait>,
}

#[bindings::export(Class)]
#[derive(Clone)]
pub struct TextToSpeechSession {
    holder: Arc<Mutex<InstanceHolder>>,
    state: Arc<Mutex<TextToSpeechSessionState>>,
}

impl TextToSpeechSession {
    pub async fn new(
        backend: Arc<dyn Backend>,
        model: Model,
        path: Option<String>,
    ) -> Result<Self, TextToSpeechSessionError> {
        if !model.specializations.contains(&ModelSpecialization::TextToSpeech {}) {
            return Err(TextToSpeechSessionError::UnsupportedModel {});
        }
        let reference = path.unwrap_or_else(|| model.identifier.clone());

        let holder = tokio::spawn(async move {
            let text_to_speech_backend =
                backend.as_text_to_speech_capable().ok_or(TextToSpeechSessionError::UnsupportedModel {})?;
            let instance = text_to_speech_backend.instance(reference, ()).await.map_err(|error| {
                TextToSpeechSessionError::Backend {
                    message: error.to_string(),
                }
            })?;
            let instance_state = instance.state().await.map_err(|error| TextToSpeechSessionError::Backend {
                message: error.to_string(),
            })?;
            Ok(InstanceHolder {
                instance,
                state: instance_state,
            })
        })
        .await
        .map_err(|error| TextToSpeechSessionError::Backend {
            message: error.to_string(),
        })??;

        Ok(Self {
            holder: Arc::new(Mutex::new(holder)),
            state: Arc::new(Mutex::new(TextToSpeechSessionState::Idle)),
        })
    }
}

#[bindings::export(Implementation)]
impl TextToSpeechSession {
    #[bindings::export(Method(Getter))]
    pub async fn state(&self) -> TextToSpeechSessionState {
        *self.state.lock().await
    }

    #[bindings::export(Method)]
    pub async fn synthesize(
        &self,
        input: String,
    ) -> Result<TextToSpeechOutput, TextToSpeechSessionError> {
        let stream = self.synthesize_stream(input).await;
        let mut outputs: Vec<TextToSpeechOutput> = Vec::new();
        while let Some(event) = stream.next().await {
            match event {
                TextToSpeechSessionStreamChunk::Output {
                    output,
                } => outputs.push(output),
                TextToSpeechSessionStreamChunk::Error {
                    error,
                } => return Err(error),
            }
        }
        let last_output = outputs.last().ok_or(TextToSpeechSessionError::NoResponse {})?;
        let pcm_batch = PcmBatch {
            channels: last_output.pcm_batch.channels,
            sample_rate: last_output.pcm_batch.sample_rate,
            lengths: vec![outputs.iter().flat_map(|output| output.pcm_batch.lengths.iter().copied()).sum()],
            samples: outputs.iter().flat_map(|output| output.pcm_batch.samples.iter().copied()).collect(),
        };
        Ok(TextToSpeechOutput {
            pcm_batch,
            stats: last_output.stats.clone(),
        })
    }

    #[bindings::export(Method)]
    pub async fn synthesize_stream(
        &self,
        input: String,
    ) -> TextToSpeechSessionStream {
        let cancel_token_to_return = CancelToken::new();
        let (sender, receiver) = mpsc::unbounded_channel::<Result<TextToSpeechOutput, TextToSpeechSessionError>>();

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
                        let _ = sender.send(Err(TextToSpeechSessionError::UnableToPerformOperationInCurrentState {}));
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

                let text_length = input.len() as u32;
                let generation_start_time = Instant::now();
                let mut first_chunk_seconds: Option<f64> = None;
                let mut audio_duration = 0.0;

                let mut stream = instance.stream(&input, backend_state.as_mut(), (), cancel_token);
                while let Some(event) = stream.next().await {
                    let item = event
                        .map(|pcm_batch| {
                            let first_chunk_seconds = *first_chunk_seconds
                                .get_or_insert_with(|| generation_start_time.elapsed().as_secs_f64());
                            audio_duration += pcm_batch.duration();
                            TextToSpeechOutput {
                                pcm_batch,
                                stats: TextToSpeechStats {
                                    text_length,
                                    first_chunk_seconds,
                                    generation_duration: generation_start_time.elapsed().as_secs_f64(),
                                    audio_duration,
                                },
                            }
                        })
                        .map_err(|error| TextToSpeechSessionError::Backend {
                            message: error.to_string(),
                        });
                    if sender.send(item).is_err() {
                        break;
                    }
                }
            }

            *state.lock().await = TextToSpeechSessionState::Idle;
        });

        TextToSpeechSessionStream {
            receiver: Arc::new(Mutex::new(receiver)),
            cancel_token: cancel_token_to_return,
        }
    }
}
