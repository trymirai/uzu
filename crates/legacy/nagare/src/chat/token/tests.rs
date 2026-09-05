use std::{
    any::Any,
    collections::VecDeque,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

use futures::Stream;
use hanashi::{
    Encoding as _,
    chat::{
        Encoding,
        hanashi::{HanashiEncodingImpl, config::HanashiConfig},
    },
};
use shoji::{
    traits::{
        State,
        backend::{Error, Instance, InstanceStream, chat_token},
    },
    types::{
        basic::{SamplingParameters, ToolDescription, ToolFunction, ToolNamespace, Value},
        session::chat::{ChatContentBlock, ChatMessage, ChatReply, ChatReplyConfig, ChatReplyFinishReason},
    },
};
use tokenizers::{AddedToken, Tokenizer, decoders::fuse::Fuse, models::bpe::BPE};
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tokio_util::sync::CancellationToken;

use super::Session;
use crate::{
    chat::{ChatSession, ChatSessionState, Instance as SessionInstance},
    telemetry::Telemetry,
};

const END_TOKEN_ID: u32 = 248046;

#[derive(Default)]
struct Audit {
    state_attempts: usize,
    state_creations: usize,
    submissions: Vec<(usize, Vec<u64>, Vec<u64>)>,
    completed: Vec<Vec<u64>>,
}

struct TestState {
    identifier: usize,
    tokens: Vec<u64>,
}

impl State for TestState {}

struct TestInstance {
    tokenizer: Arc<Tokenizer>,
    audit: Arc<Mutex<Audit>>,
    turns: Mutex<VecDeque<VecDeque<Result<chat_token::StreamOutput, Error>>>>,
    state_failures: Mutex<usize>,
    context_limit: Mutex<Option<usize>>,
}

impl Instance for TestInstance {
    type StreamConfig = ChatReplyConfig;
    type StreamInput = chat_token::StreamInput;
    type StreamOutput = chat_token::StreamOutput;
    type StreamMetrics = chat_token::StreamMetrics;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn State>, Error>> + Send + '_>> {
        Box::pin(async move {
            let mut audit = self.audit.lock().unwrap();
            audit.state_attempts += 1;
            let mut failures = self.state_failures.lock().unwrap();
            if *failures != 0 {
                *failures -= 1;
                return Err("injected state creation error".into());
            }
            audit.state_creations += 1;
            Ok(Box::new(TestState {
                identifier: audit.state_creations,
                tokens: Vec::new(),
            }) as Box<dyn State>)
        })
    }

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        state: &'a mut dyn State,
        _config: Self::StreamConfig,
        _cancel_token: CancellationToken,
    ) -> Pin<Box<dyn InstanceStream<Item = Result<Self::StreamOutput, Error>, Metrics = Self::StreamMetrics> + Send + 'a>>
    {
        let state = (state as &mut dyn Any).downcast_mut::<TestState>().unwrap();
        self.audit.lock().unwrap().submissions.push((state.identifier, input.clone(), state.tokens.clone()));
        state.tokens.extend_from_slice(input);
        Box::pin(TestStream {
            state,
            audit: self.audit.clone(),
            events: self.turns.lock().unwrap().pop_front().expect("missing test generation"),
            returned: Vec::new(),
        })
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        None
    }
}

impl chat_token::Instance for TestInstance {
    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }

    fn max_context_length(&self) -> Option<usize> {
        *self.context_limit.lock().unwrap()
    }

    fn stop_token_ids(&self) -> Option<Box<[u64]>> {
        Some(Box::new([u64::from(END_TOKEN_ID)]))
    }

    fn sampling_defaults(&self) -> SamplingParameters {
        SamplingParameters::default()
    }
}

struct TestStream<'a> {
    state: &'a mut TestState,
    audit: Arc<Mutex<Audit>>,
    events: VecDeque<Result<chat_token::StreamOutput, Error>>,
    returned: Vec<u64>,
}

impl Stream for TestStream<'_> {
    type Item = Result<chat_token::StreamOutput, Error>;

    fn poll_next(
        self: Pin<&mut Self>,
        _context: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let event = this.events.pop_front();
        if let Some(Ok(chat_token::StreamOutput::Token(token))) = &event {
            this.returned.push(*token);
        }
        Poll::Ready(event)
    }
}

impl InstanceStream for TestStream<'_> {
    type Metrics = chat_token::StreamMetrics;

    fn metrics(&self) -> Self::Metrics {
        None
    }
}

impl Drop for TestStream<'_> {
    fn drop(&mut self) {
        // Match the backend lifecycle: emitted tokens become durable when its stream is dropped.
        self.state.tokens.extend_from_slice(&self.returned);
        self.audit.lock().unwrap().completed.push(self.state.tokens.clone());
    }
}

fn tokenizer() -> Arc<Tokenizer> {
    let mut vocabulary = serde_json::Map::new();
    for character in 0..128u32 {
        vocabulary.insert(char::from_u32(character).unwrap().to_string(), serde_json::json!(character));
    }
    vocabulary.insert("ab".to_string(), serde_json::json!(128));
    vocabulary.insert("abc".to_string(), serde_json::json!(129));
    let framing = [
        ("<|im_start|>", 130),
        ("<|im_end|>", END_TOKEN_ID),
        ("<think>", 131),
        ("</think>", 132),
        ("<tool_call>", 133),
        ("</tool_call>", 134),
        ("<tool_response>", 135),
        ("</tool_response>", 136),
    ];
    for (token, identifier) in framing {
        vocabulary.insert(token.to_string(), serde_json::json!(identifier));
    }
    let roles = ["system", "developer", "assistant", "user", "tool"];
    for (index, role) in roles.iter().enumerate() {
        vocabulary.insert((*role).to_string(), serde_json::json!(137 + index));
    }
    let model: BPE = serde_json::from_str(
        &serde_json::json!({
            "type": "BPE",
            "vocab": vocabulary,
            "merges": [["a", "b"], ["ab", "c"]],
        })
        .to_string(),
    )
    .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(Fuse::new()));
    tokenizer.add_special_tokens(&framing.map(|(token, _)| AddedToken::from(token, true)));
    tokenizer.add_tokens(&roles.map(|role| AddedToken::from(role, false)));
    Arc::new(tokenizer)
}

fn encode(
    tokenizer: &Tokenizer,
    text: &str,
) -> Vec<u64> {
    tokenizer.encode(text, false).unwrap().get_ids().iter().copied().map(u64::from).collect()
}

async fn session(generations: &[&str]) -> (ChatSession, Arc<TestInstance>) {
    let tokenizer = tokenizer();
    let instance = Arc::new(TestInstance {
        tokenizer: tokenizer.clone(),
        audit: Arc::new(Mutex::new(Audit::default())),
        turns: Mutex::new(
            generations
                .iter()
                .map(|text| {
                    encode(&tokenizer, text)
                        .into_iter()
                        .map(|token| Ok(chat_token::StreamOutput::Token(token)))
                        .collect()
                })
                .collect(),
        ),
        state_failures: Mutex::new(0),
        context_limit: Mutex::new(None),
    });
    let token_session = Session {
        instance: instance.clone(),
        state: instance.state().await.unwrap(),
        encoding: Encoding::Hanashi(HanashiEncodingImpl::new(HanashiConfig::Qwen35, tokenizer).unwrap()),
        input_tokens: Vec::new(),
        stop_token_ids: Box::new([u64::from(END_TOKEN_ID)]),
        state_is_valid: true,
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        energy_recorder: super::EnergyRecorder::new(),
    };
    (
        ChatSession {
            instance: Arc::new(AsyncMutex::new(SessionInstance::Token(token_session))),
            state: Arc::new(AsyncMutex::new(ChatSessionState::Idle)),
            messages: Arc::new(AsyncMutex::new(Vec::new())),
            model_id: "prefix-cache-test".to_string(),
            telemetry: Telemetry::disabled(),
            tool_registry: None,
        },
        instance,
    )
}

fn tool_prompt() -> Vec<ChatMessage> {
    vec![
        ChatMessage::developer().with_tool_namespaces(vec![ToolNamespace {
            name: "functions".to_string(),
            description: None,
            tools: vec![ToolDescription::Function {
                tool_function: ToolFunction {
                    name: "write_file".to_string(),
                    description: "Write a file".to_string(),
                    parameters: Some(Value {
                        json: r#"{"type":"object","properties":{"content":{"type":"object"}}}"#.to_string(),
                    }),
                    return_definition: None,
                },
            }],
        }]),
        ChatMessage::user().with_text("Write the config".to_string()),
    ]
}

async fn turn(
    session: &ChatSession,
    messages: Vec<ChatMessage>,
    config: ChatReplyConfig,
    cancel_token: CancellationToken,
) -> ChatReply {
    let (sender, _receiver) = mpsc::unbounded_channel();
    let replies = session.send_input(&sender, messages, config, cancel_token, &mut Vec::new()).await;
    replies.expect("generation should succeed").pop().expect("generation should produce a reply")
}

async fn token_ledger(session: &ChatSession) -> Vec<u64> {
    let instance = session.instance.lock().await;
    let SessionInstance::Token(session) = &*instance else {
        panic!("expected token session");
    };
    session.encoding.state().tokens.iter().map(|token| u64::from(token.id)).collect()
}

#[tokio::test]
async fn compact_json_tool_reply_reuses_actual_tokens_after_stream_drop() {
    let (session, instance) = session(&[
        "\n</think>\n\n<tool_call>\n<function=write_file>\n<parameter=content>\n{\"strict\":true}\n</parameter>\n</function>\n</tool_call><|im_end|>",
        "\n</think>\n\nDone<|im_end|>",
    ]).await;
    let first = turn(&session, tool_prompt(), ChatReplyConfig::default(), CancellationToken::new()).await;
    assert_eq!(first.finish_reason, Some(ChatReplyFinishReason::ToolCalls));
    let first_tokens = token_ledger(&session).await;
    assert_eq!(instance.audit.lock().unwrap().completed, vec![first_tokens.clone()]);
    let result = ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
        identifier: None,
        name: None,
        value: Value {
            json: "\"written\"".to_string(),
        },
    });
    let second = turn(&session, vec![result], ChatReplyConfig::default(), CancellationToken::new()).await;
    let expected_suffix = encode(
        &instance.tokenizer,
        "\n<|im_start|>user\n<tool_response>\nwritten\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n",
    );
    let final_tokens = token_ledger(&session).await;
    let audit = instance.audit.lock().unwrap();
    assert_eq!(audit.state_creations, 1, "valid tool continuation must not allocate fresh backend state");
    assert_eq!(audit.submissions[1].0, audit.submissions[0].0);
    assert_eq!(audit.submissions[1].1, expected_suffix);
    assert_eq!(audit.submissions[1].2, first_tokens);
    assert!(final_tokens.starts_with(&first_tokens));
    assert_eq!(audit.completed[1], final_tokens);
    assert_eq!(second.stats.tokens_count_input_cached, Some(first_tokens.len() as u32));
    assert_eq!(second.stats.tokens_count_input, Some(expected_suffix.len() as u32));
}

#[tokio::test]
async fn sampled_token_ids_survive_a_text_equivalent_continuation() {
    let (session, instance) = session(&["\n</think>\n\n", "\n</think>\n\nDone<|im_end|>"]).await;
    {
        let mut turns = instance.turns.lock().unwrap();
        let first = turns.front_mut().unwrap();
        first.extend(
            [u64::from(b'a'), u64::from(b'b'), u64::from(b'c'), u64::from(END_TOKEN_ID)]
                .map(|token| Ok(chat_token::StreamOutput::Token(token))),
        );
    }
    assert_ne!(encode(&instance.tokenizer, "abc"), vec![u64::from(b'a'), u64::from(b'b'), u64::from(b'c')]);
    turn(
        &session,
        vec![ChatMessage::user().with_text("Say abc".to_string())],
        ChatReplyConfig::default(),
        CancellationToken::new(),
    )
    .await;
    let first_tokens = token_ledger(&session).await;
    let second = turn(
        &session,
        vec![ChatMessage::user().with_text("Continue".to_string())],
        ChatReplyConfig::default(),
        CancellationToken::new(),
    )
    .await;
    let final_tokens = token_ledger(&session).await;
    let audit = instance.audit.lock().unwrap();
    assert_eq!(audit.state_creations, 1);
    assert_eq!(audit.submissions[1].2, first_tokens);
    assert!(final_tokens.starts_with(&first_tokens), "the codec must retain the IDs that the backend consumed");
    assert_eq!(audit.completed[1], final_tokens);
    assert_eq!(second.stats.tokens_count_input_cached, Some(first_tokens.len() as u32));
}

async fn assert_next_turn_has_fresh_state(
    session: &ChatSession,
    instance: &TestInstance,
) {
    let reply = turn(
        session,
        vec![ChatMessage::user().with_text("Continue".to_string())],
        ChatReplyConfig::default(),
        CancellationToken::new(),
    )
    .await;
    let tokens = token_ledger(session).await;
    let audit = instance.audit.lock().unwrap();
    assert_eq!(audit.state_creations, 2);
    assert_eq!(audit.submissions.len(), 2);
    assert!(audit.submissions[1].2.is_empty(), "recovery must not submit a full prompt to stale backend state");
    assert_eq!(audit.completed[1], tokens);
    assert_eq!(reply.stats.tokens_count_input_cached, Some(0));
}

#[tokio::test]
async fn invalid_input_cannot_leave_empty_codec_with_reusable_backend_state() {
    let (session, instance) = session(&["\n</think>\n\nReady<|im_end|>", "\n</think>\n\nDone<|im_end|>"]).await;
    turn(
        &session,
        vec![ChatMessage::user().with_text("Start".to_string())],
        ChatReplyConfig::default(),
        CancellationToken::new(),
    )
    .await;
    let history = session.messages.lock().await.clone();
    let (sender, _receiver) = mpsc::unbounded_channel();
    let result = session
        .send_input(
            &sender,
            vec![ChatMessage::developer().with_text("Invalid late developer message".to_string())],
            ChatReplyConfig::default(),
            CancellationToken::new(),
            &mut Vec::new(),
        )
        .await;
    assert!(result.is_none());
    *session.messages.lock().await = history;
    assert_next_turn_has_fresh_state(&session, &instance).await;
}

#[tokio::test]
async fn cancelled_turn_requires_fresh_backend_state() {
    let (session, instance) = session(&["\n</think>\n\nInterrupted<|im_end|>", "\n</think>\n\nDone<|im_end|>"]).await;
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let reply = turn(
        &session,
        vec![ChatMessage::user().with_text("Start".to_string())],
        ChatReplyConfig::default(),
        cancellation,
    )
    .await;
    assert_eq!(reply.finish_reason, Some(ChatReplyFinishReason::Cancelled));
    assert_next_turn_has_fresh_state(&session, &instance).await;
}

#[tokio::test]
async fn length_limited_turn_requires_fresh_backend_state() {
    let (session, instance) = session(&["\n</think>\n\nInterrupted<|im_end|>", "\n</think>\n\nDone<|im_end|>"]).await;
    let reply = turn(
        &session,
        vec![ChatMessage::user().with_text("Start".to_string())],
        ChatReplyConfig::default().with_token_limit(Some(1)),
        CancellationToken::new(),
    )
    .await;
    assert_eq!(reply.finish_reason, Some(ChatReplyFinishReason::Length));
    assert_next_turn_has_fresh_state(&session, &instance).await;
}

#[tokio::test]
async fn backend_error_requires_fresh_backend_state() {
    let (session, instance) = session(&["\n", "\n</think>\n\nDone<|im_end|>"]).await;
    instance.turns.lock().unwrap().front_mut().unwrap().push_back(Err("injected backend error".into()));
    let (sender, _receiver) = mpsc::unbounded_channel();
    let result = session
        .send_input(
            &sender,
            vec![ChatMessage::user().with_text("Start".to_string())],
            ChatReplyConfig::default(),
            CancellationToken::new(),
            &mut Vec::new(),
        )
        .await;
    assert!(result.is_none());
    assert_next_turn_has_fresh_state(&session, &instance).await;
}

#[tokio::test]
async fn dropped_receiver_requires_fresh_backend_state() {
    let (session, instance) = session(&["\n</think>\n\nInterrupted<|im_end|>", "\n</think>\n\nDone<|im_end|>"]).await;
    let (sender, receiver) = mpsc::unbounded_channel();
    drop(receiver);
    let result = session
        .send_input(
            &sender,
            vec![ChatMessage::user().with_text("Start".to_string())],
            ChatReplyConfig::default(),
            CancellationToken::new(),
            &mut Vec::new(),
        )
        .await;
    assert!(result.is_none());
    assert_next_turn_has_fresh_state(&session, &instance).await;
}

#[tokio::test]
async fn changed_history_requires_fresh_backend_state() {
    let (session, instance) = session(&["\n</think>\n\nReady<|im_end|>", "\n</think>\n\nDone<|im_end|>"]).await;
    turn(
        &session,
        vec![ChatMessage::user().with_text("Start".to_string())],
        ChatReplyConfig::default(),
        CancellationToken::new(),
    )
    .await;
    session.messages.lock().await[0] = ChatMessage::user().with_text("An edited instruction".to_string());
    assert_next_turn_has_fresh_state(&session, &instance).await;
}

#[tokio::test]
async fn failed_backend_state_creation_retries_before_submitting_input() {
    let (session, instance) = session(&["\n</think>\n\nReady<|im_end|>", "\n</think>\n\nDone<|im_end|>"]).await;
    turn(
        &session,
        vec![ChatMessage::user().with_text("Start".to_string())],
        ChatReplyConfig::default(),
        CancellationToken::new(),
    )
    .await;
    session.messages.lock().await[0] = ChatMessage::user().with_text("An edited instruction".to_string());
    *instance.state_failures.lock().unwrap() = 1;
    let (sender, _receiver) = mpsc::unbounded_channel();
    let result = session
        .send_input(
            &sender,
            vec![ChatMessage::user().with_text("Continue".to_string())],
            ChatReplyConfig::default(),
            CancellationToken::new(),
            &mut Vec::new(),
        )
        .await;
    assert!(result.is_none());
    assert_eq!(instance.audit.lock().unwrap().submissions.len(), 1);

    let reply = turn(&session, Vec::new(), ChatReplyConfig::default(), CancellationToken::new()).await;
    let tokens = token_ledger(&session).await;
    let audit = instance.audit.lock().unwrap();
    assert_eq!(audit.state_attempts, 3);
    assert_eq!(audit.state_creations, 2);
    assert_eq!(audit.submissions.len(), 2);
    assert!(audit.submissions[1].2.is_empty());
    assert_eq!(audit.completed[1], tokens);
    assert_eq!(reply.stats.tokens_count_input_cached, Some(0));
}

#[tokio::test]
async fn failed_explicit_reset_cannot_reuse_the_previous_backend_state() {
    let (session, instance) = session(&["\n</think>\n\nReady<|im_end|>", "\n</think>\n\nDone<|im_end|>"]).await;
    turn(
        &session,
        vec![ChatMessage::user().with_text("Start".to_string())],
        ChatReplyConfig::default(),
        CancellationToken::new(),
    )
    .await;
    *instance.state_failures.lock().unwrap() = 1;
    {
        let mut instance = session.instance.lock().await;
        let SessionInstance::Token(session) = &mut *instance else {
            panic!("expected token session");
        };
        assert!(session.reset().await.is_err());
    }
    assert_next_turn_has_fresh_state(&session, &instance).await;
    assert_eq!(instance.audit.lock().unwrap().state_attempts, 3);
}

#[tokio::test]
async fn context_limited_turn_requires_fresh_backend_state() {
    let (session, instance) = session(&["\n</think>\n\nInterrupted<|im_end|>", "\n</think>\n\nDone<|im_end|>"]).await;
    *instance.context_limit.lock().unwrap() = Some(1);
    let reply = turn(
        &session,
        vec![ChatMessage::user().with_text("Start".to_string())],
        ChatReplyConfig::default(),
        CancellationToken::new(),
    )
    .await;
    assert_eq!(reply.finish_reason, Some(ChatReplyFinishReason::ContextLimitReached));
    *instance.context_limit.lock().unwrap() = None;
    assert_next_turn_has_fresh_state(&session, &instance).await;
}

#[tokio::test]
async fn incomplete_tool_call_at_stop_requires_fresh_backend_state() {
    let (session, instance) = session(&["\n</think>\n\n<tool_call><|im_end|>", "\n</think>\n\nDone<|im_end|>"]).await;
    let reply = turn(&session, tool_prompt(), ChatReplyConfig::default(), CancellationToken::new()).await;
    assert_eq!(reply.finish_reason, Some(ChatReplyFinishReason::ToolCalls));
    assert!(reply.message.content.iter().any(|block| matches!(block, ChatContentBlock::ToolCallCandidate { .. })));
    {
        let instance = session.instance.lock().await;
        let SessionInstance::Token(session) = &*instance else {
            panic!("expected token session");
        };
        assert!(!session.state_is_valid, "an incomplete tool call must not certify a completed cache");
    }
    // The malformed call cannot be rendered as a valid historical tool call. Retry with it removed.
    session.messages.lock().await.pop();
    assert_next_turn_has_fresh_state(&session, &instance).await;
}
