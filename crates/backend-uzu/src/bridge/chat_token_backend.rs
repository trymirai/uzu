use std::{
    any::Any,
    path::PathBuf,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures::Stream;
use parking_lot::{Mutex, MutexGuard};
use shoji::{
    traits::{
        State,
        backend::{
            Error as BackendError, Instance as BackendInstance, InstanceStream, NoMetricsStream,
            chat_token::{
                Instance as ChatTokenBackendInstance, StreamInput as ChatTokenStreamInput,
                StreamMetrics as ChatTokenStreamMetrics, StreamOutput as ChatTokenStreamOutput, TokenStreamOutput,
            },
        },
    },
    types::session::chat::{ChatConfig, ChatReplyConfig},
};
use tokenizers::Tokenizer;
use tokio_util::sync::CancellationToken;

use crate::{
    backends::common::Backend,
    bridge::{
        chat_token_state::UzuChatTokenBackendInstanceState,
        helpers::{error_stream, get_grammar, get_max_context_length, get_sampling_method, get_speculator},
    },
    engine::{
        Engine,
        language_model::{
            LanguageModel,
            state::LanguageModelState,
            stream::{LanguageModelStream, LanguageModelStreamOptions, LanguageModelStreamSpeculatorOptions},
        },
    },
    speculators::speculator::Speculator,
};

pub struct UzuChatTokenBackendInstance<B: Backend> {
    engine: Arc<Mutex<Engine<B>>>,
    model: Arc<Mutex<LanguageModel<B>>>,
    config: ChatConfig,
    tokenizer: Option<Tokenizer>,
    stop_token_ids: Vec<i32>,
    speculator: Option<(Box<dyn Speculator>, usize)>,
    max_context_length: Option<usize>,
}

impl<B: Backend> UzuChatTokenBackendInstance<B> {
    pub fn new(
        model_path: String,
        config: ChatConfig,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<Self, BackendError> {
        let engine = Engine::<B>::new().map_err(|err| err.to_string())?;
        let model_path = PathBuf::from(model_path);
        let model = engine.load_language_model(&model_path).map_err(|err| err.to_string())?;

        let stop_token_ids = model.generation_config().stop_token_ids.iter().map(|id| *id as i32).collect();

        let speculator = if let Some(preset) = config.speculation_preset.as_ref() {
            get_speculator(preset, tokenizer)?
        } else {
            None
        };

        let max_context_length = get_max_context_length(&model, config.context_length.clone());

        Ok(Self {
            engine: Arc::new(Mutex::new(engine)),
            model: Arc::new(Mutex::new(model)),
            config,
            tokenizer: tokenizer.map(|tok| tok.clone()),
            stop_token_ids,
            speculator,
            max_context_length,
        })
    }
}

impl<B: Backend> BackendInstance for UzuChatTokenBackendInstance<B> {
    type StreamConfig = ChatReplyConfig;
    type StreamInput = ChatTokenStreamInput;
    type StreamOutput = ChatTokenStreamOutput;
    type StreamMetrics = ChatTokenStreamMetrics;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn State>, BackendError>> + Send + '_>> {
        Box::pin(async move {
            let model = self.model.lock();
            let max_context_length = get_max_context_length(&model, self.config.context_length.clone());
            model
                .create_empty_state(max_context_length)
                .map_err(|err| BackendError::from(err.to_string()))
                .map(|state| Box::new(UzuChatTokenBackendInstanceState::new(state)) as Box<dyn State>)
        })
    }

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        state: &'a mut dyn State,
        config: Self::StreamConfig,
        cancel_token: CancellationToken,
    ) -> Pin<
        Box<
            dyn InstanceStream<Item = Result<Self::StreamOutput, BackendError>, Metrics = Self::StreamMetrics>
                + Send
                + 'a,
        >,
    > {
        let model = self.model.clone();
        let model_guard = Box::pin(model.lock());

        let state =
            (state as &mut dyn Any).downcast_mut::<UzuChatTokenBackendInstanceState<B>>().unwrap().value.clone();
        let mut state_guard = Box::pin(state.lock());

        let token_limit = config.token_limit.map(|count| count as usize);

        let grammar = if let Some(grammar_config) = config.grammar {
            if let Some(ref tokenizer) = self.tokenizer {
                match get_grammar(grammar_config, tokenizer, &self.stop_token_ids) {
                    Ok(grammar) => Some(grammar),
                    Err(err) => {
                        return Box::pin(NoMetricsStream::new(error_stream(err.to_string())));
                    },
                }
            } else {
                None
            }
        } else {
            None
        };

        let options = LanguageModelStreamOptions {
            sampling_method: get_sampling_method::<B>(&model_guard, &config.sampling_policy),
            grammar,
            speculator: self.speculator.as_ref().map(|(speculator, budget)| LanguageModelStreamSpeculatorOptions {
                speculator: speculator.as_ref(),
                speculation_budget: *budget,
                trie_creation_config: Default::default(),
            }),
        };

        let stream = match model_guard.stream(input, &mut state_guard, options) {
            Ok(iter) => iter,
            Err(err) => {
                return Box::pin(NoMetricsStream::new(error_stream(err.to_string())));
            },
        };

        Box::pin(UzuChatTokenStream::<B> {
            cancel_token: cancel_token.child_token(),
            stream: unsafe { std::mem::transmute::<LanguageModelStream<'_, B>, LanguageModelStream<'a, B>>(stream) },
            tokens_generated: 0,
            token_limit,
            _state_guard: unsafe {
                std::mem::transmute::<
                    Pin<Box<MutexGuard<'_, LanguageModelState<B>>>>,
                    Pin<Box<MutexGuard<'a, LanguageModelState<B>>>>,
                >(state_guard)
            },
            _state: state,
            _model_guard: unsafe {
                std::mem::transmute::<
                    Pin<Box<MutexGuard<'_, LanguageModel<B>>>>,
                    Pin<Box<MutexGuard<'a, LanguageModel<B>>>>,
                >(model_guard)
            },
            _model: model,
        })
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        self.engine.lock().peak_memory_usage()
    }
}

impl<B: Backend> ChatTokenBackendInstance for UzuChatTokenBackendInstance<B> {
    fn max_context_length(&self) -> Option<usize> {
        self.max_context_length
    }

    fn stop_token_ids(&self) -> Option<Box<[u64]>> {
        Some(self.stop_token_ids.iter().map(|id| *id as u64).collect())
    }
}

// Horrible code

struct UzuChatTokenStream<'a, B: Backend> {
    cancel_token: CancellationToken,
    stream: LanguageModelStream<'a, B>,
    tokens_generated: usize,
    token_limit: Option<usize>,
    _state_guard: Pin<Box<MutexGuard<'a, LanguageModelState<B>>>>,
    _state: Arc<Mutex<LanguageModelState<B>>>,
    _model_guard: Pin<Box<MutexGuard<'a, LanguageModel<B>>>>,
    _model: Arc<Mutex<LanguageModel<B>>>,
}

impl<'a, B: Backend> UzuChatTokenStream<'a, B> {
    fn next(&mut self) -> Result<Option<TokenStreamOutput>, BackendError> {
        if self.cancel_token.is_cancelled() {
            return Ok(None);
        }

        if self.token_limit.is_some_and(|token_limit| self.tokens_generated >= token_limit) {
            self.cancel_token.cancel();
            return Ok(Some(TokenStreamOutput::LimitReached));
        }

        let token = self
            .stream
            .next()
            .transpose()
            .map_err(|err| Box::<dyn std::error::Error + Send + Sync>::from(err.to_string()))?;

        if token.is_some() {
            self.tokens_generated += 1;
        }

        Ok(token.map(TokenStreamOutput::Token))
    }
}

impl<'a, B: Backend> Stream for UzuChatTokenStream<'a, B> {
    type Item = Result<TokenStreamOutput, BackendError>;

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Result<TokenStreamOutput, BackendError>>> {
        let self_mut = self.get_mut();
        let result = self_mut.next();
        if result.is_err() {
            self_mut.cancel_token.cancel();
        }
        Poll::Ready(result.transpose())
    }
}

impl<'a, B: Backend> InstanceStream for UzuChatTokenStream<'a, B> {
    type Metrics = ChatTokenStreamMetrics;

    fn metrics(&self) -> Self::Metrics {
        Some(self.stream.metrics().clone())
    }
}
