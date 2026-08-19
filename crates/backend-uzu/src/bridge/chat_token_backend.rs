use std::{
    any::Any,
    path::PathBuf,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};

use futures::Stream;
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

#[cfg(grammar)]
use crate::bridge::helpers::get_grammar;
use crate::{
    backends::common::Backend,
    bridge::{
        chat_token_state::UzuChatTokenBackendInstanceState,
        helpers::{error_stream, get_max_context_length, get_sampling_method},
    },
    engine::{
        Engine,
        language_model::{
            LanguageModel,
            stream::{LanguageModelStream, LanguageModelStreamOptions},
        },
    },
};

pub struct UzuChatTokenBackendInstance<B: Backend> {
    engine: Arc<Engine<B>>,
    model: Arc<LanguageModel<B>>,
    config: ChatConfig,
    stop_token_ids: Vec<i32>,
    max_context_length: Option<u32>,
}

impl<B: Backend> UzuChatTokenBackendInstance<B> {
    pub fn new(
        model_path: String,
        config: ChatConfig,
    ) -> Result<Self, BackendError> {
        let engine = Engine::<B>::new().map_err(|err| err.to_string())?;
        let model_path = PathBuf::from(model_path);
        let model = engine.load_language_model(&model_path).map_err(|err| err.to_string())?;

        let stop_token_ids = model.generation_config().stop_token_ids.iter().map(|id| *id as i32).collect();
        let max_context_length = get_max_context_length(&model, config.context_length.clone());

        Ok(Self {
            engine,
            model: Arc::new(model),
            config,
            stop_token_ids,
            max_context_length,
        })
    }
}

impl<B: Backend + 'static> BackendInstance for UzuChatTokenBackendInstance<B> {
    type StreamConfig = ChatReplyConfig;
    type StreamInput = ChatTokenStreamInput;
    type StreamOutput = ChatTokenStreamOutput;
    type StreamMetrics = ChatTokenStreamMetrics;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn State>, BackendError>> + Send + '_>> {
        Box::pin(async move {
            let max_context_length = get_max_context_length(&self.model, self.config.context_length.clone());
            self.model
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

        let state =
            (state as &mut dyn Any).downcast_mut::<UzuChatTokenBackendInstanceState<B>>().unwrap().value.clone();
        let state_guard = state.lock_arc();

        let token_limit = config.token_limit.map(|count| count as usize);

        #[cfg(grammar)]
        let grammar = if let Some(grammar_config) = config.grammar {
            match get_grammar(grammar_config, self.model.tokenizer(), &self.stop_token_ids) {
                Ok(grammar) => Some(grammar),
                Err(err) => {
                    return Box::pin(NoMetricsStream::new(error_stream(err.to_string())));
                },
            }
        } else {
            None
        };
        #[cfg(not(grammar))]
        if config.grammar.is_some() {
            return Box::pin(NoMetricsStream::new(error_stream("Grammar is not supported by this build".to_string())));
        }

        let options = LanguageModelStreamOptions {
            sampling_method: get_sampling_method::<B>(&self.model, &config.sampling_policy),
            #[cfg(grammar)]
            grammar,
        };

        let stream = match LanguageModelStream::new_owned(model, input, state_guard, options) {
            Ok(iter) => iter,
            Err(err) => {
                return Box::pin(NoMetricsStream::new(error_stream(err.to_string())));
            },
        };

        Box::pin(UzuChatTokenStream::<B> {
            cancel_token: cancel_token.child_token(),
            stream,
            tokens_generated: 0,
            token_limit,
            prefill_duration: None,
            decode_duration: Duration::ZERO,
        })
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        self.engine.peak_memory_usage()
    }
}

impl<B: Backend + 'static> ChatTokenBackendInstance for UzuChatTokenBackendInstance<B> {
    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.model.tokenizer().clone()
    }

    fn max_context_length(&self) -> Option<usize> {
        self.max_context_length.map(|max_context_length| max_context_length as usize)
    }

    fn stop_token_ids(&self) -> Option<Box<[u64]>> {
        Some(self.stop_token_ids.iter().map(|id| *id as u64).collect())
    }
}

struct UzuChatTokenStream<B: Backend + 'static> {
    cancel_token: CancellationToken,
    stream: LanguageModelStream<'static, B>,
    tokens_generated: usize,
    token_limit: Option<usize>,
    prefill_duration: Option<Duration>,
    decode_duration: Duration,
}

impl<B: Backend + 'static> UzuChatTokenStream<B> {
    fn next(&mut self) -> Result<Option<TokenStreamOutput>, BackendError> {
        if self.cancel_token.is_cancelled() {
            return Ok(None);
        }

        if self.token_limit.is_some_and(|token_limit| self.tokens_generated >= token_limit) {
            self.cancel_token.cancel();
            return Ok(Some(TokenStreamOutput::LimitReached));
        }

        let started = Instant::now();
        let token = self
            .stream
            .next()
            .transpose()
            .map_err(|err| Box::<dyn std::error::Error + Send + Sync>::from(err.to_string()))?;
        let elapsed = started.elapsed();
        if self.prefill_duration.is_none() {
            self.prefill_duration = Some(elapsed);
        } else {
            self.decode_duration += elapsed;
        }

        if token.is_some() {
            self.tokens_generated += 1;
        }

        Ok(token.map(TokenStreamOutput::Token))
    }
}

impl<B: Backend + 'static> Stream for UzuChatTokenStream<B> {
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

impl<B: Backend + 'static> InstanceStream for UzuChatTokenStream<B> {
    type Metrics = ChatTokenStreamMetrics;

    fn metrics(&self) -> Self::Metrics {
        let mut metrics = self.stream.metrics().clone();
        metrics.prefill_duration = self.prefill_duration;
        metrics.decode_duration = Some(self.decode_duration);
        Some(metrics)
    }
}
