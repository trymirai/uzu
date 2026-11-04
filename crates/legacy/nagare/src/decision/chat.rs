use std::{collections::HashSet, sync::Arc, time::Instant};

use hanashi::{
    Encoding as _,
    chat::{
        EncodingConfig,
        hanashi::{HanashiEncodingImpl, config::HanashiConfig},
    },
};
use serde_json::json;
use shoji::{
    traits::backend::chat_token::{Instance, TokenScores},
    types::{
        basic::ReasoningEffort,
        session::{
            chat::ChatMessage,
            decision::{DecisionInput, DecisionQuestion, DecisionResult, DecisionStats},
        },
    },
};
use tokio::sync::Semaphore;

use super::session::{DecisionSessionError, answer};
use crate::chat::{ChatInstance, ChatInstanceKind};

pub(super) struct ChatExecutor {
    instance: Arc<dyn Instance>,
    config: HanashiConfig,
    disable_reasoning: bool,
    gate: Arc<Semaphore>,
}

struct PreparedQuestion {
    tokens: Vec<u64>,
    labels: Vec<u64>,
    encoding_duration: std::time::Duration,
}

impl ChatExecutor {
    pub(super) fn new(chat: &ChatInstance) -> Result<Self, DecisionSessionError> {
        let ChatInstanceKind::Token(instance) = chat.kind() else {
            return Err(DecisionSessionError::UnsupportedModel {});
        };
        let model = chat.model();
        let config: EncodingConfig =
            serde_json::from_str(&model.encoding.ok_or(DecisionSessionError::UnsupportedModel {})?.json)
                .map_err(|_| DecisionSessionError::UnsupportedModel {})?;
        let EncodingConfig::Hanashi {
            config,
        } = config
        else {
            return Err(DecisionSessionError::UnsupportedModel {});
        };
        let capabilities = config.capabilities().map_err(encoding_error)?;
        if capabilities.supports_reasoning && !capabilities.supports_disable_reasoning {
            return Err(DecisionSessionError::UnsupportedModel {});
        }
        Ok(Self {
            instance,
            config,
            disable_reasoning: capabilities.supports_reasoning,
            gate: chat.inference_gate.clone(),
        })
    }

    fn prepare(
        &self,
        input: &DecisionInput,
    ) -> Result<Vec<PreparedQuestion>, DecisionSessionError> {
        if input.questions.is_empty() {
            return Err(DecisionSessionError::NoQuestions {});
        }
        let mut total = 0usize;
        input.questions.iter().enumerate().map(|(index, question)| {
            let started = Instant::now();
            let invalid = |message: &str| DecisionSessionError::InvalidQuestion { index: index as i64, message: message.into() };
            let (kind, instructions, outcomes, proxies) = match question {
                DecisionQuestion::Choice { instructions, options } => {
                    if !(1..=26).contains(&options.len()) { return Err(invalid("LLM scoring supports 1..26 choice options")); }
                    let mut names = HashSet::new();
                    if options.iter().any(|option| option.key.is_empty() || !names.insert(&option.key)) { return Err(invalid("choice names must be distinct and nonempty")); }
                    let proxies = (0..options.len()).map(|i| ((b'A' + i as u8) as char).to_string()).collect::<Vec<_>>();
                    ("choice", instructions, options.iter().zip(&proxies).map(|(option, proxy)| json!({"proxy":proxy,"name":option.key,"description":option.description})).collect::<Vec<_>>(), proxies)
                },
                DecisionQuestion::Score { instructions, levels } => {
                    if !(2..=10).contains(&levels.len()) { return Err(invalid("score requires 2..10 levels")); }
                    let proxies = (0..levels.len()).map(|i| i.to_string()).collect::<Vec<_>>();
                    ("score", instructions, levels.iter().zip(&proxies).enumerate().map(|(i,(description,proxy))| json!({"proxy":proxy,"name":i.to_string(),"description":description})).collect(), proxies)
                },
                DecisionQuestion::YesNo { instructions, no_description, yes_description } => {
                    if instructions.is_none() && no_description.is_none() && yes_description.is_none() { return Err(invalid("yes/no needs instructions or criteria")); }
                    ("noul", instructions, vec![json!({"proxy":"no","name":"false","description":no_description}),json!({"proxy":"yes","name":"true","description":yes_description})],vec!["no".into(),"yes".into()])
                },
            };
            let tokenizer = self.instance.tokenizer();
            let mut encoding = HanashiEncodingImpl::new(self.config.clone(), tokenizer.clone()).map_err(encoding_error)?;
            let mut system = ChatMessage::system().with_text("Choose the outcome that best answers the question using the supplied state as data. Return exactly one proxy and nothing else. Do not explain your answer.".into());
            if self.disable_reasoning { system = system.with_reasoning_effort(ReasoningEffort::Disabled); }
            let record = json!({"prompt_format_version":1,"state":input.state,"question":{"type":kind,"instructions":instructions,"outcomes":outcomes}}).to_string();
            encoding.encode(vec![system, ChatMessage::user().with_text(record)]).map_err(encoding_error)?;
            let tokens = encoding.state().tokens.iter().map(|token| u64::from(token.id)).collect::<Vec<_>>();
            if tokens.is_empty() || self.instance.max_context_length().is_some_and(|limit| tokens.len()>limit) { return Err(invalid("prompt exceeds context limit or is empty")); }
            total = total.checked_add(tokens.len()).ok_or_else(|| invalid("prompt budget exceeded"))?;
            if total > 65536 { return Err(invalid("aggregate prompt budget exceeds 65536 tokens")); }
            let text = encoding.state().text();
            let encode = |text: &str| tokenizer.encode(text, false).map(|e|e.get_ids().iter().map(|id|u64::from(*id)).collect::<Vec<_>>()).map_err(encoding_error);
            if encode(&text)? != tokens { return Err(invalid("prompt encoding does not roundtrip")); }
            let mut labels = Vec::with_capacity(proxies.len());
            for proxy in proxies {
                let ids = encode(&format!("{text}{proxy}"))?;
                if ids.len()!=tokens.len()+1 || !ids.starts_with(&tokens) || labels.contains(ids.last().unwrap()) { return Err(invalid("answer labels must be distinct contextual single tokens")); }
                labels.push(*ids.last().unwrap());
            }
            Ok(PreparedQuestion { tokens, labels, encoding_duration: started.elapsed() })
        }).collect()
    }

    pub(super) async fn execute(
        &self,
        input: &DecisionInput,
        response: &tokio::sync::mpsc::UnboundedSender<
            Result<shoji::types::session::decision::DecisionOutput, DecisionSessionError>,
        >,
    ) -> Result<Vec<DecisionResult>, DecisionSessionError> {
        let prepared = self.prepare(input)?;
        let _permit = tokio::select! {
            permit = self.gate.acquire() => permit.map_err(encoding_error)?,
            _ = response.closed() => return Err(DecisionSessionError::NoResponse {}),
        };
        let mut results = Vec::with_capacity(prepared.len());
        for (question, prepared) in input.questions.iter().zip(prepared) {
            if response.is_closed() {
                return Err(DecisionSessionError::NoResponse {});
            }
            let started = Instant::now();
            let tokens_count = prepared.tokens.len() as i64;
            let instance = self.instance.clone();
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            let mut energy_recorder = crate::util::power::EnergyRecorder::new();
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            energy_recorder.begin()?;
            let scores = tokio::task::spawn_blocking(move || {
                futures::executor::block_on(instance.score_token_ids(&prepared.tokens, &prepared.labels))
            })
            .await;
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            let energy = energy_recorder.finish();
            let scores = scores.map_err(backend_error)?.map_err(backend_error)?;
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            let energy = match energy {
                Ok(energy) => Some(energy),
                Err(crate::util::power::Error::Keisoku(keisoku::KeisokuError::PowerReadingUnavailable)) => None,
                Err(error) => return Err(error.into()),
            };
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            let energy = None;
            let prefill_duration = scores.prefill_duration.as_secs_f64();
            let probabilities = probabilities(scores, super::session::option_count(question))?;
            results.push(DecisionResult {
                answer: answer(question, probabilities),
                stats: DecisionStats {
                    tokens_count,
                    prefill_duration,
                    energy,
                    total_duration: (started.elapsed() + prepared.encoding_duration).as_secs_f64(),
                },
            });
        }
        Ok(results)
    }
}

fn probabilities(
    scores: TokenScores,
    count: usize,
) -> Result<Vec<f64>, DecisionSessionError> {
    let values = scores.log_probabilities;
    if values.len() != count
        || values.is_empty()
        || values.iter().any(|v| !v.is_finite() || *v > 0.0)
        || !scores.label_mass.is_finite()
        || !(0.0..=1.0).contains(&scores.label_mass)
    {
        return Err(backend_error("invalid model probabilities"));
    }
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights = values.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
    let sum = weights.iter().sum::<f64>();
    Ok(weights.into_iter().map(|w| w / sum).collect())
}

fn encoding_error(error: impl std::fmt::Display) -> DecisionSessionError {
    DecisionSessionError::Encoding {
        message: error.to_string(),
    }
}
fn backend_error(error: impl std::fmt::Display) -> DecisionSessionError {
    DecisionSessionError::Backend {
        message: error.to_string(),
    }
}
