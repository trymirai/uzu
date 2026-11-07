use std::{
    any::Any,
    path::PathBuf,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures::{Stream, stream};
use shoji::{
    traits::{
        State,
        backend::{
            Error as ShojiBackendError, Instance as BackendInstance, InstanceStream, NoMetricsStream,
            chat_token::{
                Instance as ChatTokenBackendInstance, StreamInput as ChatTokenStreamInput,
                StreamMetrics as ChatTokenStreamMetrics, StreamOutput as ChatTokenStreamOutput, TokenStreamMetrics,
                TokenStreamOutput,
            },
        },
    },
    types::{
        basic::{BackendError, SamplingParameters, SamplingSeed},
        session::chat::{ChatConfig, ChatReplyConfig},
    },
};
use tokenizers::Tokenizer;
use tokio_util::sync::CancellationToken;
use uzu_engine::{
    backends::Backend,
    engine::{Engine, language_model::LanguageModel},
};

use crate::engine::{
    MemoryCheck,
    bridge::{
        chat_token_state::UzuChatTokenBackendInstanceState,
        helpers::{
            error_to_stream_output, get_grammar, get_max_context_length, get_sampling_method,
            grammar_trigger_token_sequence, grammar_trigger_token_sequence_for_prompt, stream_error_to_backend_error,
        },
    },
};

#[derive(Clone)]
pub struct UzuChatTokenBackendInstance<B: Backend> {
    engine: Arc<Engine<B>>,
    model: Arc<LanguageModel<B>>,
    config: ChatConfig,
    stop_token_ids: Vec<i32>,
    max_context_length: Option<u32>,
    sampling_defaults: SamplingParameters,
    grammar_trigger_token_sequence: Option<Vec<u64>>,
}

impl<B: Backend> UzuChatTokenBackendInstance<B> {
    /// Refuses a model that fails `memory_check` against the memory free on the device before loading its weights.
    pub async fn new(
        model_path: String,
        config: ChatConfig,
        memory_check: Option<MemoryCheck>,
    ) -> Result<Self, ShojiBackendError> {
        let engine = Engine::<B>::new().await.map_err(|err| err.to_string())?;
        if let Some(shortfall) =
            memory_check.zip(engine.available_memory()).and_then(|(check, available)| check(available as u64))
        {
            return Err(shortfall.to_string().into());
        }
        let model_path = PathBuf::from(model_path);
        let model = engine.load_language_model(&model_path).await.map_err(|err| err.to_string())?;

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
        let grammar_trigger_token_sequence =
            grammar_trigger_token_sequence(model.end_of_thinking_tag(), model.tokenizer());

        Ok(Self {
            engine,
            model: Arc::new(model),
            config,
            stop_token_ids,
            max_context_length,
            sampling_defaults,
            grammar_trigger_token_sequence,
        })
    }
}

impl<B: Backend> BackendInstance for UzuChatTokenBackendInstance<B> {
    type StreamConfig = ChatReplyConfig;
    type StreamInput = ChatTokenStreamInput;
    type StreamOutput = ChatTokenStreamOutput;
    type StreamMetrics = ChatTokenStreamMetrics;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn State>, ShojiBackendError>> + Send + '_>> {
        Box::pin(async move {
            let sampling_seed = match &self.config.sampling_seed {
                SamplingSeed::Default {} => rand::random(),
                SamplingSeed::Custom {
                    seed,
                } => *seed as u64,
            };
            self.model
                .create_empty_state(self.max_context_length, sampling_seed)
                .map_err(|err| ShojiBackendError::from(err.to_string()))
                .map(|state| Box::new(UzuChatTokenBackendInstanceState::new(state, sampling_seed)) as Box<dyn State>)
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
            dyn InstanceStream<Item = Result<Self::StreamOutput, ShojiBackendError>, Metrics = Self::StreamMetrics>
                + Send
                + 'a,
        >,
    > {
        let state = (state as &mut dyn Any).downcast_mut::<UzuChatTokenBackendInstanceState<B>>().unwrap();

        let token_limit = config.token_limit.map(|count| count as usize);

        let grammar = if let Some(grammar_config) = config.grammar {
            let trigger_token_sequence = grammar_trigger_token_sequence_for_prompt(
                self.grammar_trigger_token_sequence.as_deref(),
                input,
                self.model.tokenizer(),
            );
            match get_grammar(grammar_config, self.model.tokenizer(), &self.stop_token_ids, trigger_token_sequence) {
                Ok(grammar) => Some(grammar),
                Err(err) => {
                    return Box::pin(NoMetricsStream::new(error_to_stream_output(BackendError::Grammar {
                        message: err.to_string(),
                    })));
                },
            }
        } else {
            None
        };

        let model_state = state.value.get_mut();
        model_state.set_sampling_seed(match config.sampling_seed {
            SamplingSeed::Default {} => state.sampling_seed,
            SamplingSeed::Custom {
                seed,
            } => seed as u64,
        });

        let sampling_method = match get_sampling_method::<B>(&self.model, &config.sampling_policy) {
            Ok(sampling_method) => sampling_method,
            Err(err) => {
                return Box::pin(NoMetricsStream::new(error_to_stream_output(BackendError::other(err))));
            },
        };

        let mut options = self.model.default_stream_options();
        options.sampling_method = sampling_method;
        options.grammar = grammar;

        let stream = match self.model.stream(input, model_state, options) {
            Ok(stream) => stream,
            Err(err) => {
                return Box::pin(NoMetricsStream::new(error_to_stream_output(stream_error_to_backend_error(err))));
            },
        };

        let metrics = stream.metrics().clone();
        let inner = stream::unfold(
            (Some(stream), 0, cancel_token.child_token()),
            move |(stream, mut tokens_generated, cancel_token)| async move {
                let mut stream = stream?;
                let (mut output, finished) = if cancel_token.is_cancelled() {
                    (None, true)
                } else if token_limit.is_some_and(|token_limit| tokens_generated >= token_limit) {
                    (Some(Ok(TokenStreamOutput::LimitReached)), true)
                } else {
                    match stream
                        .generate()
                        .await
                        .map_err(|error| ShojiBackendError::from(stream_error_to_backend_error(error)))
                    {
                        Ok(Some(token)) => {
                            tokens_generated += 1;
                            (Some(Ok(TokenStreamOutput::Token(token))), self.stop_token_ids.contains(&(token as i32)))
                        },
                        Ok(None) => (None, true),
                        Err(error) => (Some(Err(error)), true),
                    }
                };
                if finished {
                    cancel_token.cancel();
                    if let Err(error) = stream.close().await
                        && !matches!(&output, Some(Err(_)))
                    {
                        output = Some(Err(stream_error_to_backend_error(error).into()));
                    }
                }
                let metrics = stream.metrics().clone();
                output
                    .map(|output| ((output, metrics), ((!finished).then_some(stream), tokens_generated, cancel_token)))
            },
        );

        Box::pin(UzuChatTokenStream {
            inner: Box::pin(inner),
            metrics,
        })
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        self.engine.peak_memory_usage()
    }
}

impl<B: Backend> ChatTokenBackendInstance for UzuChatTokenBackendInstance<B> {
    fn with_config(
        &self,
        config: ChatConfig,
    ) -> Box<dyn ChatTokenBackendInstance> {
        Box::new(Self {
            max_context_length: get_max_context_length(&self.model, config.context_length.clone()),
            config,
            ..self.clone()
        })
    }

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

    fn speculation_notice(&self) -> Option<String> {
        self.model
            .speculator_unavailable()
            .then(|| format!("Speculation is off: the {} backend cannot run this model's speculator", B::NAME))
    }
}

struct UzuChatTokenStream<S> {
    inner: Pin<Box<S>>,
    metrics: TokenStreamMetrics,
}

impl<S> Stream for UzuChatTokenStream<S>
where
    S: Stream<Item = (Result<TokenStreamOutput, ShojiBackendError>, TokenStreamMetrics)>,
{
    type Item = Result<TokenStreamOutput, ShojiBackendError>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<TokenStreamOutput, ShojiBackendError>>> {
        let self_mut = self.get_mut();
        self_mut.inner.as_mut().poll_next(cx).map(|item| {
            item.map(|(output, metrics)| {
                self_mut.metrics = metrics;
                output
            })
        })
    }
}

impl<S> InstanceStream for UzuChatTokenStream<S>
where
    S: Stream<Item = (Result<TokenStreamOutput, ShojiBackendError>, TokenStreamMetrics)>,
{
    type Metrics = ChatTokenStreamMetrics;

    fn metrics(&self) -> Self::Metrics {
        Some(self.metrics.clone())
    }
}
