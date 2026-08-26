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
    types::{
        basic::{SamplingParameters, SamplingSeed},
        session::chat::{ChatConfig, ChatReplyConfig},
    },
};
use tokenizers::Tokenizer;
use tokio_util::sync::CancellationToken;
use uzu_engine::{
    backends::common::Backend,
    engine::{
        Engine,
        language_model::{LanguageModel, state::LanguageModelState, stream::LanguageModelStream},
    },
};

#[cfg(feature = "capability-grammar")]
use crate::engine::bridge::helpers::{
    get_grammar, grammar_trigger_token_sequence, grammar_trigger_token_sequence_for_prompt,
};
use crate::engine::bridge::{
    chat_token_state::UzuChatTokenBackendInstanceState,
    helpers::{error_stream, get_max_context_length, get_sampling_method},
};

pub struct UzuChatTokenBackendInstance<B: Backend> {
    engine: Arc<Engine<B>>,
    model: Arc<LanguageModel<B>>,
    config: ChatConfig,
    stop_token_ids: Vec<i32>,
    max_context_length: Option<u32>,
    sampling_defaults: SamplingParameters,
    #[cfg(feature = "capability-grammar")]
    grammar_trigger_token_sequence: Option<Vec<u64>>,
}

impl<B: Backend> UzuChatTokenBackendInstance<B> {
    pub fn new(
        model_path: String,
        config: ChatConfig,
    ) -> Result<Self, BackendError> {
        let engine = Engine::<B>::new().map_err(|err| err.to_string())?;
        let model_path = PathBuf::from(model_path);
        let model = engine.load_language_model(&model_path).map_err(|err| err.to_string())?;

        let generation_config = model.generation_config();
        let stop_token_ids = generation_config.stop_token_ids.iter().map(|id| *id as i32).collect();
        let sampling_defaults = SamplingParameters {
            temperature: generation_config.temperature.map(|value| value as f64),
            top_k: generation_config.top_k.map(|value| value as i64),
            top_p: generation_config.top_p.map(|value| value as f64),
            min_p: generation_config.min_p.map(|value| value as f64),
            repetition_penalty: generation_config.repetition_penalty.map(|value| value as f64),
            suffix_repetition_length: generation_config.suffix_repetition_length.map(|value| value as i64),
        };
        let max_context_length = get_max_context_length(&model, config.context_length.clone());
        #[cfg(feature = "capability-grammar")]
        let grammar_trigger_token_sequence = grammar_trigger_token_sequence(&model);

        Ok(Self {
            engine,
            model: Arc::new(model),
            config,
            stop_token_ids,
            max_context_length,
            sampling_defaults,
            #[cfg(feature = "capability-grammar")]
            grammar_trigger_token_sequence,
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
            let max_context_length = get_max_context_length(&self.model, self.config.context_length.clone());
            let sampling_seed = match &self.config.sampling_seed {
                SamplingSeed::Default {} => rand::random(),
                SamplingSeed::Custom {
                    seed,
                } => *seed as u64,
            };
            self.model
                .create_empty_state(max_context_length, sampling_seed)
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

        let state =
            (state as &mut dyn Any).downcast_mut::<UzuChatTokenBackendInstanceState<B>>().unwrap().value.clone();
        let mut state_guard = Box::pin(state.lock());

        let token_limit = config.token_limit.map(|count| count as usize);

        #[cfg(feature = "capability-grammar")]
        let grammar = if let Some(grammar_config) = config.grammar {
            let trigger_token_sequence = grammar_trigger_token_sequence_for_prompt(
                self.grammar_trigger_token_sequence.as_deref(),
                input,
                self.model.tokenizer(),
            );
            match get_grammar(grammar_config, self.model.tokenizer(), &self.stop_token_ids, trigger_token_sequence) {
                Ok(grammar) => Some(grammar),
                Err(err) => {
                    return Box::pin(NoMetricsStream::new(error_stream(err.to_string())));
                },
            }
        } else {
            None
        };
        #[cfg(not(feature = "capability-grammar"))]
        if config.grammar.is_some() {
            return Box::pin(NoMetricsStream::new(error_stream("Grammar is not supported by this build".to_string())));
        }

        let mut options = self.model.default_stream_options();
        options.sampling_method = get_sampling_method::<B>(&self.model, &config.sampling_policy);
        #[cfg(feature = "capability-grammar")]
        {
            options.grammar = grammar;
        }

        let stream = match self.model.stream(input, &mut state_guard, options) {
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
            _model: model,
        })
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        self.engine.peak_memory_usage()
    }
}

impl<B: Backend> ChatTokenBackendInstance for UzuChatTokenBackendInstance<B> {
    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.model.tokenizer().clone()
    }

    fn max_context_length(&self) -> Option<usize> {
        self.max_context_length.map(|max_context_length| max_context_length as usize)
    }

    fn stop_token_ids(&self) -> Option<Box<[u64]>> {
        Some(self.stop_token_ids.iter().map(|id| *id as u64).collect())
    }

    fn sampling_defaults(&self) -> SamplingParameters {
        self.sampling_defaults
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
    _model: Arc<LanguageModel<B>>,
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
