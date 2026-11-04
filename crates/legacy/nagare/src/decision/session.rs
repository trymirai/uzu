#[cfg(test)]
#[path = "../../unit/decision/session_test.rs"]
mod tests;

use std::{sync::Arc, time::Instant};

use hanashi::{DecisionEncoding, DecisionEncodingError};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use keisoku::KeisokuError;
use serde::{Deserialize, Serialize};
use shoji::{
    traits::{Backend, backend::decision::DecisionInstance},
    types::{
        model::{Model, ModelSpecialization},
        session::decision::{
            DecisionAnswer, DecisionInput, DecisionOutput, DecisionQuestion, DecisionResult, DecisionStats,
        },
    },
};
use tokio::sync::{Mutex, mpsc};

pub use crate::decision::error::DecisionSessionError;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::util::power::{EnergyRecorder, Error as EnergyError};

#[bindings::export(Enumeration)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecisionSessionState {
    Idle,
    Deciding,
}

struct DecisionInstanceHolder {
    instance: Box<dyn DecisionInstance>,
    encoding: DecisionEncoding,
}

enum Executor {
    D3(DecisionInstanceHolder),
    Chat(super::chat::ChatExecutor),
}

pub(super) struct PreparedQuestion<'a> {
    pub(super) tokens: Vec<u64>,
    pub(super) scorer: Scorer<'a>,
    pub(super) encoding_duration: std::time::Duration,
}

pub(super) enum Scorer<'a> {
    D3 {
        instance: &'a dyn DecisionInstance,
        option_count: usize,
    },
    Chat {
        instance: Arc<dyn shoji::traits::backend::chat_token::Instance>,
        labels: Vec<u64>,
    },
}

impl PreparedQuestion<'_> {
    async fn score(self) -> Result<(Vec<f64>, std::time::Duration), DecisionSessionError> {
        let error = |error: String| DecisionSessionError::Backend {
            message: error,
        };
        match self.scorer {
            Scorer::D3 {
                instance,
                option_count,
            } => {
                let output =
                    instance.probabilities(&self.tokens, option_count).await.map_err(|e| error(e.to_string()))?;
                Ok((output.probabilities.into_iter().map(f64::from).collect(), output.prefill_duration))
            },
            Scorer::Chat {
                instance,
                labels,
            } => {
                let count = labels.len();
                let scores = tokio::task::spawn_blocking(move || {
                    futures::executor::block_on(instance.score_token_ids(&self.tokens, &labels))
                })
                .await
                .map_err(|e| error(e.to_string()))?
                .map_err(|e| error(e.to_string()))?;
                let duration = scores.prefill_duration;
                Ok((super::chat::probabilities(scores, count)?, duration))
            },
        }
    }
}

#[bindings::export(Class)]
#[derive(Clone)]
pub struct DecisionSession {
    holder: Arc<Mutex<Executor>>,
    state: Arc<Mutex<DecisionSessionState>>,
}

impl DecisionSession {
    pub async fn new(
        backend: Arc<dyn Backend>,
        model: Model,
        path: Option<String>,
    ) -> Result<Self, DecisionSessionError> {
        if !model.specializations.contains(&ModelSpecialization::Decision {}) {
            return Err(DecisionSessionError::UnsupportedModel {});
        }
        let reference = path.unwrap_or_else(|| model.identifier.clone());

        let holder = tokio::spawn(async move {
            let decision_backend = backend.as_decision_capable().ok_or(DecisionSessionError::UnsupportedModel {})?;
            let instance =
                decision_backend.instance(reference, ()).await.map_err(|error| DecisionSessionError::Backend {
                    message: error.to_string(),
                })?;
            let encoding = DecisionEncoding::new(instance.token_codec_config(), instance.tokenizer())
                .map_err(|_| DecisionSessionError::UnsupportedModel {})?;

            Ok::<_, DecisionSessionError>(DecisionInstanceHolder {
                instance,
                encoding,
            })
        })
        .await
        .map_err(|error| DecisionSessionError::Backend {
            message: error.to_string(),
        })??;

        Ok(Self {
            holder: Arc::new(Mutex::new(Executor::D3(holder))),
            state: Arc::new(Mutex::new(DecisionSessionState::Idle)),
        })
    }

    pub fn from_chat_instance(chat: &crate::chat::ChatInstance) -> Result<Self, DecisionSessionError> {
        Ok(Self {
            holder: Arc::new(Mutex::new(Executor::Chat(super::chat::ChatExecutor::new(chat)?))),
            state: Arc::new(Mutex::new(DecisionSessionState::Idle)),
        })
    }

    async fn execute(
        executor: &Executor,
        input: &DecisionInput,
        response: &mpsc::UnboundedSender<Result<DecisionOutput, DecisionSessionError>>,
    ) -> Result<Vec<DecisionResult>, DecisionSessionError> {
        if input.questions.is_empty() {
            return Err(DecisionSessionError::NoQuestions {});
        }

        let (prepared, gate) = match executor {
            Executor::D3(holder) => (Self::prepare_d3(holder, input)?, None),
            Executor::Chat(chat) => (chat.prepare(input)?, Some(chat.gate.clone())),
        };
        let _guard = match gate.as_ref() {
            Some(gate) => Some(tokio::select! {
                guard = gate.write() => guard,
                _ = response.closed() => return Err(DecisionSessionError::NoResponse {}),
            }),
            None => None,
        };
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let mut energy_recorder = EnergyRecorder::new();
        let mut results = Vec::with_capacity(prepared.len());
        for (question, prepared) in input.questions.iter().zip(prepared) {
            if gate.is_some() && response.is_closed() {
                return Err(DecisionSessionError::NoResponse {});
            }
            let time_start = Instant::now();
            let tokens_count = prepared.tokens.len() as i64;
            let encoding_duration = prepared.encoding_duration;
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            energy_recorder.begin()?;
            let output = prepared.score().await;
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            let energy = energy_recorder.finish();
            let (probabilities, prefill_duration) = output?;
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            let energy = match energy {
                Ok(energy) => Some(energy),
                Err(EnergyError::Keisoku(KeisokuError::PowerReadingUnavailable)) => None,
                Err(error) => return Err(error.into()),
            };
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            let energy = None;
            results.push(DecisionResult {
                answer: answer(question, probabilities),
                stats: DecisionStats {
                    tokens_count,
                    prefill_duration: prefill_duration.as_secs_f64(),
                    total_duration: (encoding_duration + time_start.elapsed()).as_secs_f64(),
                    energy,
                },
            });
        }
        Ok(results)
    }

    fn prepare_d3<'a>(
        holder: &'a DecisionInstanceHolder,
        input: &DecisionInput,
    ) -> Result<Vec<PreparedQuestion<'a>>, DecisionSessionError> {
        input
            .questions
            .iter()
            .enumerate()
            .map(|(index, question)| {
                let time_start = Instant::now();
                let prompt = holder.encoding.encode(&input.state, question).map_err(|error| match error {
                    DecisionEncodingError::DuplicateOption(_) | DecisionEncodingError::InvalidQuestion(_) => {
                        DecisionSessionError::InvalidQuestion {
                            index: index as i64,
                            message: error.to_string(),
                        }
                    },
                    DecisionEncodingError::UnableToEncodeText
                    | DecisionEncodingError::UnableToRenderPrompt(_)
                    | DecisionEncodingError::UnsupportedTokenCodec => DecisionSessionError::Encoding {
                        message: error.to_string(),
                    },
                })?;
                Ok(PreparedQuestion {
                    tokens: prompt.into_iter().map(u64::from).collect(),
                    scorer: Scorer::D3 {
                        instance: holder.instance.as_ref(),
                        option_count: option_count(question),
                    },
                    encoding_duration: time_start.elapsed(),
                })
            })
            .collect()
    }
}

#[bindings::export(Implementation)]
impl DecisionSession {
    #[bindings::export(Method(Getter))]
    pub async fn state(&self) -> DecisionSessionState {
        *self.state.lock().await
    }

    #[bindings::export(Method)]
    pub async fn decide(
        &self,
        input: DecisionInput,
    ) -> Result<DecisionOutput, DecisionSessionError> {
        let (sender, mut receiver) = mpsc::unbounded_channel::<Result<DecisionOutput, DecisionSessionError>>();
        let holder = self.holder.clone();
        let state = self.state.clone();

        tokio::spawn(async move {
            {
                let mut state = state.lock().await;
                match *state {
                    DecisionSessionState::Idle => {
                        *state = DecisionSessionState::Deciding;
                    },
                    DecisionSessionState::Deciding => {
                        let _ = sender.send(Err(DecisionSessionError::UnableToPerformOperationInCurrentState {}));
                        return;
                    },
                }
            }

            let result = {
                let holder = holder.lock().await;
                Self::execute(&holder, &input, &sender).await
            };
            let output = result.map(|results| DecisionOutput {
                results,
            });

            *state.lock().await = DecisionSessionState::Idle;
            let _ = sender.send(output);
        });

        receiver.recv().await.unwrap_or(Err(DecisionSessionError::NoResponse {}))
    }
}

pub(super) fn option_count(question: &DecisionQuestion) -> usize {
    match question {
        DecisionQuestion::Choice {
            options,
            ..
        } => options.len(),
        DecisionQuestion::Score {
            levels,
            ..
        } => levels.len(),
        DecisionQuestion::YesNo {
            ..
        } => 2,
    }
}

// `probabilities` follow the question's option order: choice options, score levels, or [no, yes].
pub(super) fn answer(
    question: &DecisionQuestion,
    probabilities: Vec<f64>,
) -> DecisionAnswer {
    match question {
        DecisionQuestion::Choice {
            options,
            ..
        } => {
            // The first option wins a tie, as in the reference `d3_format.to_answer`.
            let index = (1..probabilities.len()).fold(0, |best, index| {
                if probabilities[index] > probabilities[best] {
                    index
                } else {
                    best
                }
            });
            DecisionAnswer::Choice {
                index: index as i64,
                key: options[index].key.clone(),
                probabilities,
            }
        },
        DecisionQuestion::Score {
            ..
        } => DecisionAnswer::Score {
            score: probabilities.iter().enumerate().map(|(level, probability)| level as f64 * probability).sum(),
            probabilities,
        },
        DecisionQuestion::YesNo {
            ..
        } => DecisionAnswer::YesNo {
            yes_probability: probabilities[1],
        },
    }
}
