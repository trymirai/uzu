use std::{pin::Pin, sync::Mutex as StdMutex};

use shoji::{
    traits::backend::{
        Error,
        decision::{Config, DecisionBackend, DecisionSessionOutput},
    },
    types::{
        model::{ModelAccessibility, ModelSource},
        session::{
            classification::{ChatTokenCodecConfig, TokenCodecConfig},
            decision::DecisionOption,
        },
    },
};
use tokenizers::{Tokenizer, models::bpe::BPE};

use super::*;

struct D3Fixture {
    calls: Arc<StdMutex<Vec<usize>>>,
}
struct D3Instance {
    calls: Arc<StdMutex<Vec<usize>>>,
}
impl Backend for D3Fixture {
    fn identifier(&self) -> String {
        "fixture".into()
    }
    fn version(&self) -> String {
        "1".into()
    }
    fn as_decision_capable(&self) -> Option<&dyn DecisionBackend> {
        Some(self)
    }
}
impl DecisionBackend for D3Fixture {
    fn instance(
        &self,
        _: String,
        _: Config,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn DecisionInstance>, Error>> + Send + '_>> {
        Box::pin(async move {
            Ok(Box::new(D3Instance {
                calls: self.calls.clone(),
            }) as Box<dyn DecisionInstance>)
        })
    }
}
impl DecisionInstance for D3Instance {
    fn tokenizer(&self) -> Arc<Tokenizer> {
        Arc::new(Tokenizer::new(BPE::builder().vocab_and_merges([("x".into(), 0)], Vec::new()).build().unwrap()))
    }
    fn token_codec_config(&self) -> TokenCodecConfig {
        TokenCodecConfig::Chat(ChatTokenCodecConfig {
            prompt_template: "{{ messages[0].content.state }}".into(),
            output_parser_regex: None,
            system_role_name: "system".into(),
            user_role_name: "user".into(),
            assistant_role_name: "assistant".into(),
            eos_token: None,
            bos_token: None,
            end_of_thinking_tag: None,
            default_system_prompt: None,
        })
    }
    fn codes(&self) -> Vec<String> {
        vec!["A".into(), "B".into()]
    }
    fn probabilities<'a>(
        &'a self,
        input: &'a [u64],
        count: usize,
    ) -> Pin<Box<dyn Future<Output = Result<DecisionSessionOutput, Error>> + Send + 'a>> {
        Box::pin(async move {
            assert!(!input.is_empty());
            self.calls.lock().unwrap().push(count);
            Ok(DecisionSessionOutput {
                probabilities: vec![0.25, 0.75],
                prefill_duration: std::time::Duration::from_millis(3),
                memory_used_bytes: Some(1024),
            })
        })
    }
}

#[tokio::test]
async fn d3_session_keeps_existing_encoding_and_answer_aggregation() {
    let calls = Arc::new(StdMutex::new(Vec::new()));
    let backend = Arc::new(D3Fixture {
        calls: calls.clone(),
    });
    let mut model = Model::external(
        "d3".into(),
        "fixture".into(),
        "D3".into(),
        "fixture".into(),
        "Fixture".into(),
        "1".into(),
        vec![],
        ModelAccessibility::OnDevice {
            source: ModelSource::Filesystem {
                path: String::new(),
            },
        },
        None,
    );
    model.specializations = vec![ModelSpecialization::Decision {}];
    let session = DecisionSession::new(backend, model, None).await.unwrap();
    let output = session
        .decide(DecisionInput {
            state: "x".into(),
            questions: vec![
                DecisionQuestion::Choice {
                    instructions: None,
                    options: vec![DecisionOption::from("a", "A"), DecisionOption::from("b", "B")],
                },
                DecisionQuestion::Score {
                    instructions: None,
                    levels: vec!["low".into(), "high".into()],
                },
                DecisionQuestion::YesNo {
                    instructions: Some("yes?".into()),
                    no_description: None,
                    yes_description: None,
                },
            ],
        })
        .await
        .unwrap();
    assert_eq!(*calls.lock().unwrap(), vec![2, 2, 2]);
    assert!(output.results.iter().all(|result| result.stats.prefill_duration == 0.003));
    assert_eq!(
        output.results[0].answer,
        DecisionAnswer::Choice {
            index: 1,
            key: "b".into(),
            probabilities: vec![0.25, 0.75]
        }
    );
    assert_eq!(
        output.results[1].answer,
        DecisionAnswer::Score {
            score: 0.75,
            probabilities: vec![0.25, 0.75]
        }
    );
    assert_eq!(
        output.results[2].answer,
        DecisionAnswer::YesNo {
            yes_probability: 0.75
        }
    );
    assert_eq!(session.state().await, DecisionSessionState::Idle);
}
