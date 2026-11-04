use std::{
    pin::Pin,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use hanashi::chat::{EncodingConfig, hanashi::config::HanashiConfig};
use shoji::{
    traits::backend::{
        Error, Instance as BackendInstance, InstanceStream, State,
        chat_token::{Instance, StreamInput, StreamOutput, TokenScoreError, TokenScores, TokenStreamMetrics},
    },
    types::{
        basic::SamplingParameters,
        model::{ModelAccessibility, ModelSource},
        session::{
            chat::ChatReplyConfig,
            decision::{DecisionAnswer, DecisionInput, DecisionQuestion},
        },
    },
};
use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::decision::DecisionSession;

struct Numeric {
    tokenizer: Arc<Tokenizer>,
    calls: AtomicUsize,
    prompts: Mutex<Vec<Vec<u64>>>,
    started: tokio::sync::Notify,
    release: (Mutex<bool>, std::sync::Condvar),
    limit: Option<usize>,
    supported: bool,
    malformed: Mutex<Option<Vec<f64>>>,
}
impl BackendInstance for Numeric {
    type StreamConfig = ChatReplyConfig;
    type StreamInput = StreamInput;
    type StreamOutput = StreamOutput;
    type StreamMetrics = Option<TokenStreamMetrics>;
    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn State>, Error>> + Send + '_>> {
        panic!("decisions must not allocate generation state")
    }
    fn stream<'a>(
        &'a self,
        _: &'a StreamInput,
        _: &'a mut dyn State,
        _: ChatReplyConfig,
        _: CancellationToken,
    ) -> Pin<Box<dyn InstanceStream<Item = Result<StreamOutput, Error>, Metrics = Self::StreamMetrics> + Send + 'a>>
    {
        panic!("decisions must not generate")
    }
    fn peak_memory_usage(&self) -> Option<usize> {
        None
    }
}
impl Instance for Numeric {
    fn with_config(
        &self,
        _: ChatConfig,
    ) -> Box<dyn Instance> {
        panic!("decisions must not create a second instance")
    }
    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }
    fn max_context_length(&self) -> Option<usize> {
        self.limit
    }
    fn stop_token_ids(&self) -> Option<Box<[u64]>> {
        None
    }
    fn sampling_defaults(&self) -> SamplingParameters {
        panic!("decisions must not sample")
    }
    fn score_token_ids<'a>(
        &'a self,
        prompt: &'a [u64],
        labels: &'a [u64],
    ) -> Pin<Box<dyn Future<Output = Result<TokenScores, TokenScoreError>> + Send + 'a>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.prompts.lock().unwrap().push(prompt.to_vec());
            self.started.notify_one();
            let (lock, condition) = &self.release;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = condition.wait(released).unwrap();
            }
            drop(released);
            if !self.supported {
                return Err(TokenScoreError::Unsupported);
            }
            Ok(TokenScores {
                log_probabilities: self
                    .malformed
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or_else(|| {
                        labels
                            .iter()
                            .map(|id| {
                                if self.tokenizer.decode(&[*id as u32], false).unwrap() == "yes" {
                                    0.8_f64.ln()
                                } else {
                                    0.2_f64.ln()
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .into_boxed_slice(),
                label_mass: 0.001,
                prefill_duration: std::time::Duration::from_millis(2),
            })
        })
    }
}
fn fixture(
    limit: Option<usize>,
    supported: bool,
) -> (ChatInstance, Arc<Numeric>) {
    let mut config = HanashiConfig::Qwen35.resolve().unwrap();
    let mut alphabet = ByteLevel::alphabet().into_iter().collect::<Vec<_>>();
    alphabet.sort_unstable();
    let vocab: [_; 256] = alphabet
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), i as u32))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let mut tokenizer = Tokenizer::new(BPE::builder().vocab_and_merges(vocab, Vec::new()).build().unwrap());
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    let mut markers = config.parsing.framing_config().tokens;
    markers.extend(["<|im_start|>".into(), "<|im_end|>".into(), "<think>".into(), "</think>".into()]);
    tokenizer.add_special_tokens(&markers.into_iter().map(|s| AddedToken::from(s, true)).collect::<Vec<_>>());
    tokenizer.add_tokens(&["yes", "no", "system", "user", "assistant"].map(|s| AddedToken::from(s, false)));
    config.tokens.bos_token_id = None;
    config.tokens.eos_token_id = tokenizer.token_to_id("<|im_end|>");
    let mut encoding = serde_json::to_value(EncodingConfig::Hanashi {
        config: HanashiConfig::Custom {
            config,
        },
    })
    .unwrap();
    encoding["rendering"]["jinja"]["required_functions"] = serde_json::json!([]);
    let model = Model::external(
        "test".into(),
        "test".into(),
        "Test".into(),
        "test".into(),
        "Test".into(),
        "1".into(),
        vec![],
        ModelAccessibility::OnDevice {
            source: ModelSource::Filesystem {
                path: String::new(),
            },
        },
        Some(encoding.into()),
    );
    let backend = Arc::new(Numeric {
        tokenizer: Arc::new(tokenizer),
        calls: AtomicUsize::new(0),
        prompts: Mutex::new(Vec::new()),
        started: tokio::sync::Notify::new(),
        release: (Mutex::new(true), std::sync::Condvar::new()),
        limit,
        supported,
        malformed: Mutex::new(None),
    });
    (
        ChatInstance {
            kind: ChatInstanceKind::Token(backend.clone()),
            model,
            reference: String::new(),
            inference_gate: Arc::new(tokio::sync::Semaphore::new(1)),
        },
        backend,
    )
}
fn input() -> DecisionInput {
    DecisionInput {
        state: "A receipt is present.".into(),
        questions: vec![DecisionQuestion::YesNo {
            instructions: Some("Is a receipt present?".into()),
            no_description: None,
            yes_description: None,
        }],
    }
}

#[tokio::test]
async fn llm_decisions_share_loaded_instance_without_sampling_and_preserve_yes_order() {
    let (loaded, backend) = fixture(None, true);
    let decisions = DecisionSession::from_chat_instance(&loaded).unwrap();
    let output = decisions.decide(input()).await.unwrap();
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
    let DecisionAnswer::YesNo {
        yes_probability,
    } = output.results[0].answer
    else {
        panic!()
    };
    assert!((yes_probability - 0.8).abs() < 1e-12);
    assert!(output.results[0].stats.tokens_count > 0);
    assert_eq!(output.results[0].stats.prefill_duration, 0.002);
    let second = decisions.decide(input()).await.unwrap();
    assert_eq!(second.results[0].answer, output.results[0].answer);
    let prompts = backend.prompts.lock().unwrap();
    assert_eq!(prompts[0], prompts[1]);
}

#[tokio::test]
async fn later_invalid_llm_question_prevents_all_scoring() {
    let (loaded, backend) = fixture(None, true);
    let decisions = DecisionSession::from_chat_instance(&loaded).unwrap();
    let mut request = input();
    request.questions.push(DecisionQuestion::Choice {
        instructions: None,
        options: vec![],
    });
    assert!(decisions.decide(request).await.is_err());
    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn unsupported_llm_scoring_fails_without_generation() {
    let (loaded, backend) = fixture(None, false);
    let decisions = DecisionSession::from_chat_instance(&loaded).unwrap();
    assert!(decisions.decide(input()).await.is_err());
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn queued_decision_is_cancelled_before_starting_scoring() {
    let (loaded, backend) = fixture(None, true);
    let permit = loaded.inference_gate.acquire().await.unwrap();
    let decisions = DecisionSession::from_chat_instance(&loaded).unwrap();
    let running = decisions.clone();
    let waiter = tokio::spawn(async move { running.decide(input()).await });
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while decisions.state().await != crate::decision::DecisionSessionState::Deciding {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    waiter.abort();
    let _ = waiter.await;
    drop(permit);
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while decisions.state().await != crate::decision::DecisionSessionState::Idle {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn context_overflow_fails_before_scoring() {
    let (loaded, backend) = fixture(Some(1), true);
    let decisions = DecisionSession::from_chat_instance(&loaded).unwrap();
    assert!(decisions.decide(input()).await.is_err());
    assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn invalid_llm_probabilities_are_rejected() {
    let (loaded, backend) = fixture(None, true);
    *backend.malformed.lock().unwrap() = Some(vec![f64::NAN, 0.0]);
    let decisions = DecisionSession::from_chat_instance(&loaded).unwrap();
    assert!(decisions.decide(input()).await.is_err());
}

#[tokio::test]
async fn dropped_decision_waiter_retains_permit_until_scoring_finishes() {
    let (loaded, backend) = fixture(None, true);
    *backend.release.0.lock().unwrap() = false;
    let decisions = DecisionSession::from_chat_instance(&loaded).unwrap();
    let mut request = input();
    request.questions.push(request.questions[0].clone());
    let waiter = tokio::spawn(async move { decisions.decide(request).await });
    tokio::time::timeout(std::time::Duration::from_secs(3), backend.started.notified()).await.unwrap();
    assert_eq!(loaded.inference_gate.available_permits(), 0);
    waiter.abort();
    let _ = waiter.await;
    assert_eq!(loaded.inference_gate.available_permits(), 0);
    *backend.release.0.lock().unwrap() = true;
    backend.release.1.notify_all();
    let _permit = tokio::time::timeout(std::time::Duration::from_secs(3), loaded.inference_gate.acquire())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
}
