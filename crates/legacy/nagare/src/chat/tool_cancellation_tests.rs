use std::{
    pin::Pin,
    sync::{Arc, Mutex as StdMutex},
    time::Duration,
};

use hanashi::chat::{EncodingConfig, hanashi::config::HanashiConfig};
use shoji::{
    traits::backend::{
        Error, Instance as BackendInstance, InstanceStream, NoMetricsStream, State,
        chat_token::{Instance as TokenInstance, StreamInput, StreamOutput, TokenStreamMetrics},
    },
    types::{
        basic::SamplingParameters,
        model::{ModelAccessibility, ModelSource},
    },
};
use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};

use super::*;

struct EmptyState;
impl State for EmptyState {}

// Exercise the real token session, including its cached prefix and FunctionGemma
// renderer/parser. Only the model's sampled tokens are scripted.
struct FunctionGemma {
    tokenizer: Arc<Tokenizer>,
    cached_tokens: StdMutex<Vec<u32>>,
    prompts: StdMutex<Vec<String>>,
}

impl BackendInstance for FunctionGemma {
    type StreamConfig = ChatReplyConfig;
    type StreamInput = StreamInput;
    type StreamOutput = StreamOutput;
    type StreamMetrics = Option<TokenStreamMetrics>;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn State>, Error>> + Send + '_>> {
        Box::pin(async {
            self.cached_tokens.lock().unwrap().clear();
            Ok(Box::new(EmptyState) as Box<dyn State>)
        })
    }

    fn stream<'a>(
        &'a self,
        input: &'a StreamInput,
        _state: &'a mut dyn State,
        _config: ChatReplyConfig,
        _cancel_token: CancellationToken,
    ) -> Pin<Box<dyn InstanceStream<Item = Result<StreamOutput, Error>, Metrics = Self::StreamMetrics> + Send + 'a>>
    {
        let mut cached = self.cached_tokens.lock().unwrap();
        cached.extend(input.iter().map(|id| *id as u32));
        let prompt = self.tokenizer.decode(&cached, false).unwrap();
        let output = if prompt.contains("Follow up") {
            "Answer<end_of_turn>"
        } else if prompt.contains("<end_function_response>") {
            "Continuation<end_of_turn>"
        } else {
            "<start_function_call>call:clock{}<end_function_call>"
        };
        self.prompts.lock().unwrap().push(prompt);
        let tokens = self.tokenizer.encode(output, false).unwrap().get_ids().to_vec();
        cached.extend(&tokens);
        Box::pin(NoMetricsStream::new(futures::stream::iter(
            tokens.into_iter().map(|id| Ok(StreamOutput::Token(u64::from(id)))),
        )))
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        None
    }
}

impl TokenInstance for FunctionGemma {
    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }

    fn max_context_length(&self) -> Option<usize> {
        None
    }

    fn stop_token_ids(&self) -> Option<Box<[u64]>> {
        Some(
            ["<end_function_call>", "<end_of_turn>"]
                .map(|token| u64::from(self.tokenizer.token_to_id(token).unwrap()))
                .into(),
        )
    }

    fn sampling_defaults(&self) -> SamplingParameters {
        SamplingParameters::default()
    }
}

async fn session() -> (ChatSession, Arc<FunctionGemma>) {
    let mut config = HanashiConfig::FunctionGemma.resolve().unwrap();
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let vocabulary: [(String, u32); 256] = alphabet
        .into_iter()
        .enumerate()
        .map(|(index, value)| (value.to_string(), index as u32))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let mut tokenizer = Tokenizer::new(BPE::builder().vocab_and_merges(vocabulary, Vec::new()).build().unwrap());
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    tokenizer.add_tokens(&["developer", "user", "model", "clock"].map(|value| AddedToken::from(value, false)));
    let mut markers = config.parsing.framing_config().tokens;
    markers.extend(["<bos>".to_string(), "<eos>".to_string()]);
    tokenizer.add_special_tokens(&markers.into_iter().map(|value| AddedToken::from(value, true)).collect::<Vec<_>>());
    config.tokens.bos_token_id = tokenizer.token_to_id("<bos>");
    config.tokens.eos_token_id = tokenizer.token_to_id("<eos>");
    let backend = Arc::new(FunctionGemma {
        tokenizer: Arc::new(tokenizer),
        cached_tokens: StdMutex::new(Vec::new()),
        prompts: StdMutex::new(Vec::new()),
    });
    let mut encoding = serde_json::to_value(EncodingConfig::Hanashi {
        config: HanashiConfig::Custom {
            config,
        },
    })
    .unwrap();
    // JinjaConfig omits empty lists on serialization but requires this key when read.
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
    let instance = token::Session::with_instance(backend.clone(), String::new(), &model).await.unwrap();
    let mut session = ChatSession {
        instance: Arc::new(Mutex::new(Instance::Token(instance))),
        state: Arc::new(Mutex::new(ChatSessionState::Idle)),
        messages: Arc::new(Mutex::new(Vec::new())),
        tool_registry: Some(Arc::new(Mutex::new(ToolRegistry::new()))),
    };
    session
        .add_tool(ToolDescriptor::new(
            "clock".into(),
            "Return the time".into(),
            None,
            None,
            Box::new(|_| Box::new(async { Ok(serde_json::json!("noon").into()) })),
        ))
        .await
        .unwrap();
    (session, backend)
}

async fn stop_after_tool_result(drop_receiver: bool) {
    let (session, backend) = session().await;
    let stream = session
        .reply_with_stream(vec![ChatMessage::user().with_text("Time?".into())], ChatReplyConfig::default())
        .await;
    let completed = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match stream.next().await.unwrap() {
                ChatSessionStreamChunk::ToolResults {
                    messages,
                } => break messages,
                ChatSessionStreamChunk::Error {
                    error,
                } => panic!("first reply failed: {error}"),
                ChatSessionStreamChunk::Replies {
                    ..
                } => {},
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].tool_call_results()[0].2.json, "\"noon\"");
    if drop_receiver {
        drop(stream);
    } else {
        stream.cancel_token().cancel();
        assert!(stream.next().await.is_none());
    }
    tokio::time::timeout(Duration::from_secs(5), async {
        while session.state().await != ChatSessionState::Idle {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(backend.prompts.lock().unwrap().len(), 1, "stopping must not start another prefill");
    let retained = session.messages().await;
    assert_eq!(retained.last(), completed.last());

    let replies = session
        .reply(vec![ChatMessage::user().with_text("Follow up".into())], ChatReplyConfig::default())
        .await
        .unwrap();
    assert_eq!(replies[0].message.text().as_deref(), Some("Answer"));
    let prompts = backend.prompts.lock().unwrap();
    let resumed = prompts.last().unwrap();
    assert!(resumed.contains(
        "<start_function_response>response:clock{value:<escape>noon<escape>}<end_function_response><end_of_turn>\n<start_of_turn>user\nFollow up<end_of_turn>\n<start_of_turn>model\n"
    ), "wrong interrupted-turn boundary: {resumed}");
    assert_eq!(resumed.matches("<start_function_call>").count(), 1);
    assert_eq!(resumed.matches("<start_function_response>").count(), 1);
}

#[tokio::test]
async fn cancellation_after_tool_results_keeps_history_and_allows_the_next_user_turn() {
    stop_after_tool_result(false).await;
}

#[tokio::test]
async fn dropped_receiver_after_tool_results_skips_prefill_and_allows_the_next_user_turn() {
    stop_after_tool_result(true).await;
}

#[tokio::test]
async fn normal_tool_continuation_keeps_the_existing_model_turn_open() {
    let (session, backend) = session().await;
    let replies =
        session.reply(vec![ChatMessage::user().with_text("Time?".into())], ChatReplyConfig::default()).await.unwrap();
    assert_eq!(replies[0].message.text().as_deref(), Some("Continuation"));
    let prompts = backend.prompts.lock().unwrap();
    assert_eq!(prompts.len(), 2);
    assert!(prompts[1].ends_with("<end_function_response>"));
    assert_eq!(prompts[1].matches("<start_of_turn>model\n").count(), 1);
}
