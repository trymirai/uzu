use std::{
    any::Any,
    path::PathBuf,
    pin::Pin,
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
        session::chat::ChatReplyConfig,
    },
};
use tokio_util::sync::CancellationToken;

use crate::{
    backends::common::Backend,
    bridge::container::Container,
    encodable_block::sampling::{SamplingMethod as UzuSamplingMethod, SamplingProcessingOrder},
    engine::{
        Engine,
        language_model::{
            LanguageModel,
            state::LanguageModelState,
            stream::{LanguageModelStreamError, LanguageModelStreamOptions},
        },
    },
};

pub struct UzuChatTokenLlmInstance<B: Backend> {
    model: Container<LanguageModel<B>>,
}

impl<B: Backend> UzuChatTokenLlmInstance<B> {
    pub fn new(model_path: String) -> Result<Self, Error> {
        let engine = Engine::<B>::new().map_err(|err| err.to_string())?;
        let model = engine.load_language_model(&PathBuf::from(model_path)).map_err(|err| err.to_string())?;
        Ok(Self {
            model: Container::new(model),
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
            .value
            .lock()
            .map_err(|err| Error::from(err.to_string()))
            .and_then(|model| {
                model.create_empty_state(model.recommended_context_length()).map_err(|err| Error::from(err.to_string()))
            })
            .map(|state| Box::new(Container::new(state)) as Box<dyn LlmInstanceState>);
        Box::pin(async move { result })
    }

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        state: &'a mut dyn LlmInstanceState,
        config: Self::StreamConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, Error>> + Send + 'a>> {
        let state = match state.as_any_mut().downcast_mut::<Container<LanguageModelState<B>>>() {
            Some(state) => state.clone(),
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

struct UzuChatTokenStream<B: Backend> {
    iterator: Box<dyn Iterator<Item = Result<u64, LanguageModelStreamError<B>>> + 'static>,
    state: Container<LanguageModelState<B>>,
    stop_token_ids: Box<[u64]>,
    max_length: Option<usize>,
    finished: bool,
}

impl<B: Backend> UzuChatTokenStream<B> {
    pub fn new(
        model: Container<LanguageModel<B>>,
        state: Container<LanguageModelState<B>>,
        input: &Vec<u64>,
        config: ChatReplyConfig,
    ) -> Result<Self, Error> {
        let model_guard = model.value.lock().map_err(|err| Error::from(err.to_string()))?;

        let options = Self::get_stream_options(&model_guard, config);
        let max_length = model_guard.recommended_context_length();
        let stop_token_ids = model_guard.generation_config().stop_token_ids.clone();

        let mut state_guard = state.value.lock().map_err(|err| Error::from(err.to_string()))?;
        let iterator: Box<dyn Iterator<Item = Result<u64, LanguageModelStreamError<B>>> + '_> =
            Box::new(model_guard.stream(input, &mut state_guard, options).map_err(|err| Error::from(err.to_string()))?);
        let iterator: Box<dyn Iterator<Item = Result<u64, LanguageModelStreamError<B>>> + 'static> =
            unsafe { std::mem::transmute(iterator) };

        drop(model_guard);
        drop(state_guard);

        Ok(Self {
            iterator,
            state,
            stop_token_ids,
            max_length,
            finished: false,
        })
    }

    fn get_stream_options<'a>(
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

unsafe impl<B: Backend> Send for UzuChatTokenStream<B> {}

impl<B: Backend> Stream for UzuChatTokenStream<B> {
    type Item = Result<ChatTokenStreamOutput, Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        if let Some(max_length) = self.max_length {
            let result = match self.state.value.lock() {
                Ok(state) => Ok(state.tokens().len() >= max_length),
                Err(err) => Err(Error::from(err.to_string())),
            };
            match result {
                Ok(true) => {
                    self.finished = true;
                    return Poll::Ready(None);
                },
                Ok(false) => {},
                Err(err) => {
                    self.finished = true;
                    return Poll::Ready(Some(Err(err)));
                },
            }
        }

        let item = match self.iterator.next() {
            Some(Ok(token)) => {
                if self.stop_token_ids.contains(&token) {
                    self.finished = true;
                }
                Some(Ok(token))
            },
            Some(Err(err)) => {
                self.finished = true;
                Some(Err(Error::from(err.to_string())))
            },
            None => {
                self.finished = true;
                None
            },
        };

        Poll::Ready(item)
    }
}

impl<B: Backend> LlmInstanceState for Container<LanguageModelState<B>> {
    fn clone_boxed(&self) -> Box<dyn LlmInstanceState> {
        Box::new(self.clone())
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
