use std::{
    any::Any,
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

use futures::{Stream, StreamExt, stream};
use shoji::{
    traits::{
        Instance as LlmInstance, State as LlmInstanceState,
        backend::{
            Error,
            chat_token::{StreamInput as ChatTokenStreamInput, StreamOutput as ChatTokenStreamOutput},
        },
    },
    types::{
        basic::{SamplingMethod as ShojiSamplingMethod, SamplingPolicy as ShojiSamplingPolicy},
        session::chat::{ChatConfig, ChatReplyConfig, ChatReplyFinishReason},
    },
};
use tokio_util::sync::CancellationToken;

use crate::{
    backends::common::Backend,
    encodable_block::sampling::{SamplingMethod as UzuSamplingMethod, SamplingProcessingOrder},
    engine::{
        Engine,
        language_model::{
            LanguageModel,
            state::LanguageModelState,
            stream::{LanguageModelStreamOptions, sync::LanguageModelStreamer},
        },
    },
};

pub struct UzuChatTokenLlmInstance<B: Backend> {
    model: Arc<Mutex<LanguageModel<B>>>,
}

unsafe impl<B: Backend> Send for UzuChatTokenLlmInstance<B> {}
unsafe impl<B: Backend> Sync for UzuChatTokenLlmInstance<B> {}

impl<B: Backend> UzuChatTokenLlmInstance<B> {
    pub fn new(
        model_path: String,
        config: ChatConfig,
    ) -> Result<Self, Error> {
        let engine = Engine::<B>::new().map_err(|err| err.to_string())?;
        let model = engine.load_language_model(&PathBuf::from(model_path)).map_err(|err| err.to_string())?;
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
        })
    }
}

impl<B: Backend> LlmInstance for UzuChatTokenLlmInstance<B> {
    type StreamConfig = ChatReplyConfig;
    type StreamInput = ChatTokenStreamInput;
    type StreamOutput = ChatTokenStreamOutput;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn LlmInstanceState>, Error>> + Send + '_>> {
        let result = self
            .model
            .lock()
            .map_err(|err| Error::from(err.to_string()))
            .and_then(|model| {
                model.create_empty_state(model.recommended_context_length()).map_err(|err| Error::from(err.to_string()))
            })
            .map(|state| UzuChatTokenLlmInstanceState::new(state).clone_boxed());
        Box::pin(async move { result })
    }

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        state: &'a mut dyn LlmInstanceState,
        config: Self::StreamConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, Error>> + Send + 'a>> {
        let state = match state.as_any_mut().downcast_mut::<UzuChatTokenLlmInstanceState<B>>() {
            Some(state) => state.value.clone(),
            None => return error_stream("unexpected state type for uzu chat token instance".to_string()),
        };
        match UzuChatTokenStream::new(self.model.clone(), state, input, config) {
            Ok(stream) => stream,
            Err(err) => return error_stream(err.to_string()),
        }
        .take_until(cancel_token.cancelled_owned())
        .boxed()
    }
}

struct UzuChatTokenStream<'a, B: Backend> {
    streamer: LanguageModelStreamer<'a>,
    model: Arc<Mutex<LanguageModel<B>>>,
    state: Arc<Mutex<LanguageModelState<B>>>,
    finished: bool,
}

impl<'a, B: Backend> UzuChatTokenStream<'a, B> {
    pub fn new(
        model: Arc<Mutex<LanguageModel<B>>>,
        state: Arc<Mutex<LanguageModelState<B>>>,
        input: &Vec<u64>,
        config: ChatReplyConfig,
    ) -> Result<Self, Error> {
        let model_guard = model.lock().map_err(|err| Error::from(err.to_string()))?;
        let state_guard = state.lock().map_err(|err| Error::from(err.to_string()))?;

        let options = Self::get_stream_options(&model_guard, config);
        let streamer =
            LanguageModelStreamer::new(input, &state_guard, options).map_err(|err| Error::from(err.to_string()))?;

        drop(model_guard);
        drop(state_guard);

        Ok(Self {
            streamer,
            model,
            state,
            finished: false,
        })
    }

    fn get_stream_options(
        model: &LanguageModel<B>,
        config: ChatReplyConfig,
    ) -> LanguageModelStreamOptions<'a> {
        let sampling_method = Self::get_sampling_method(model, &config.sampling_policy);
        LanguageModelStreamOptions {
            sampling_method,
            grammar: None,
            speculator: None,
        }
    }

    fn get_sampling_method(
        model: &LanguageModel<B>,
        sampling_method: &ShojiSamplingPolicy,
    ) -> UzuSamplingMethod {
        match sampling_method {
            ShojiSamplingPolicy::Default {
                ..
            } => model.default_sampling_method(),
            ShojiSamplingPolicy::Custom {
                method,
            } => match method {
                ShojiSamplingMethod::Greedy {
                    ..
                } => UzuSamplingMethod::Greedy,
                ShojiSamplingMethod::Stochastic {
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    repetition_penalty,
                    suffix_repetition_length,
                } => UzuSamplingMethod::Stochastic {
                    temperature: temperature.map(|value| value as f32),
                    top_k: top_k.map(|value| value as u32),
                    top_p: top_p.map(|value| value as f32),
                    min_p: min_p.map(|value| value as f32),
                    repetition_penalty: repetition_penalty.map(|value| value as f32),
                    suffix_repetition_length: suffix_repetition_length.map(|value| value as usize),
                    processing_order: SamplingProcessingOrder::TemperatureThenFilters,
                },
            },
        }
    }
}

unsafe impl<'a, B: Backend> Send for UzuChatTokenStream<'a, B> {}

impl<'a, B: Backend> Stream for UzuChatTokenStream<'a, B> {
    type Item = Result<ChatTokenStreamOutput, Error>;

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }
        let self_mut = self.get_mut();

        let step_result = {
            let model = match self_mut.model.lock() {
                Ok(model) => model,
                Err(err) => {
                    self_mut.finished = true;
                    return Poll::Ready(Some(Err(Error::from(err.to_string()))));
                },
            };

            if self_mut.streamer.is_stopped() {
                self_mut.finished = true;
                return Poll::Ready(None);
            }

            let mut state = match self_mut.state.lock() {
                Ok(state) => state,
                Err(err) => {
                    self_mut.finished = true;
                    return Poll::Ready(Some(Err(Error::from(err.to_string()))));
                },
            };

            if model.recommended_context_length().is_some_and(|max_length| state.tokens().len() >= max_length) {
                self_mut.finished = true;
                return Poll::Ready(Some(Ok(ChatTokenStreamOutput::Finished(
                    ChatReplyFinishReason::ContextLimitReached,
                ))));
            }

            self_mut.streamer.step(&model, &mut state).map_err(|err| Error::from(err.to_string()))
        };

        Poll::Ready(step_result.map(|token| token.map(ChatTokenStreamOutput::Token)).transpose())
    }
}

struct UzuChatTokenLlmInstanceState<B: Backend> {
    value: Arc<Mutex<LanguageModelState<B>>>,
}

impl<B: Backend> UzuChatTokenLlmInstanceState<B> {
    fn new(value: LanguageModelState<B>) -> Self {
        Self {
            value: Arc::new(Mutex::new(value)),
        }
    }
}

unsafe impl<B: Backend> Send for UzuChatTokenLlmInstanceState<B> {}
unsafe impl<B: Backend> Sync for UzuChatTokenLlmInstanceState<B> {}

impl<B: Backend> LlmInstanceState for UzuChatTokenLlmInstanceState<B> {
    fn clone_boxed(&self) -> Box<dyn LlmInstanceState> {
        Box::new(Self {
            value: self.value.clone(),
        })
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn error_stream<'a>(message: String) -> Pin<Box<dyn Stream<Item = Result<ChatTokenStreamOutput, Error>> + Send + 'a>> {
    Box::pin(stream::once(async move {
        Err::<ChatTokenStreamOutput, Error>(Box::<dyn std::error::Error + Send + Sync>::from(message))
    }))
}
