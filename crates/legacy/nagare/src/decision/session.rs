use std::{sync::Arc, time::Instant};

use hanashi::{DecisionEncoding, DecisionEncodingError};
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecisionSessionState {
    Idle,
    Deciding,
}

struct DecisionInstanceHolder {
    instance: Box<dyn DecisionInstance>,
    encoding: DecisionEncoding,
}

#[derive(Clone)]
pub struct DecisionSession {
    holder: Arc<Mutex<DecisionInstanceHolder>>,
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

            Ok(DecisionInstanceHolder {
                instance,
                encoding,
            })
        })
        .await
        .map_err(|error| DecisionSessionError::Backend {
            message: error.to_string(),
        })??;

        Ok(Self {
            holder: Arc::new(Mutex::new(holder)),
            state: Arc::new(Mutex::new(DecisionSessionState::Idle)),
        })
    }

    async fn execute(
        holder: &DecisionInstanceHolder,
        input: &DecisionInput,
    ) -> Result<Vec<DecisionResult>, DecisionSessionError> {
        if input.questions.is_empty() {
            return Err(DecisionSessionError::NoQuestions {});
        }

        // Every prompt is built before the first forward pass, so an invalid question fails the call early.
        let prompts = input
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
                Ok((prompt, time_start.elapsed()))
            })
            .collect::<Result<Vec<_>, DecisionSessionError>>()?;

        let mut results = Vec::with_capacity(prompts.len());
        for (question, (prompt, encoding_duration)) in input.questions.iter().zip(prompts) {
            let time_start = Instant::now();
            let tokens_count = prompt.len();
            let tokens = prompt.into_iter().map(u64::from).collect::<Vec<_>>();
            let probabilities =
                holder.instance.probabilities(&tokens, option_count(question)).await.map_err(|error| {
                    DecisionSessionError::Backend {
                        message: error.to_string(),
                    }
                })?;
            results.push(DecisionResult {
                answer: answer(question, probabilities),
                stats: DecisionStats {
                    tokens_count: tokens_count as i64,
                    total_duration: (encoding_duration + time_start.elapsed()).as_secs_f64(),
                },
            });
        }
        Ok(results)
    }
}

impl DecisionSession {
    pub async fn state(&self) -> DecisionSessionState {
        *self.state.lock().await
    }

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
                Self::execute(&holder, &input).await
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

fn option_count(question: &DecisionQuestion) -> usize {
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
fn answer(
    question: &DecisionQuestion,
    probabilities: Vec<f32>,
) -> DecisionAnswer {
    let probabilities = probabilities.into_iter().map(f64::from).collect::<Vec<_>>();
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
