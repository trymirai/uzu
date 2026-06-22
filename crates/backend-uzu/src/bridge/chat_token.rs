use std::{
    any::Any,
    path::PathBuf,
    pin::Pin,
    sync::MutexGuard,
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
        // TODO agolokoz: ask what this method should actually do
        Box::pin(async move {
            let model = self.model.value.lock().map_err(|err| Error::from(err.to_string()))?;
            let state = model
                .create_empty_state(model.recommended_context_length())
                .map_err(|err| Error::from(err.to_string()))?;
            Ok(Box::new(Container::new(state)) as Box<dyn LlmInstanceState>)
        })
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

        let stream = match UzuChatTokenStream::new(state, self.model.clone(), input, config) {
            Ok(stream) => stream,
            Err(err) => return error_stream(err.to_string()),
        };
        stream.take_until(cancel_token.cancelled_owned()).boxed()
    }
}

struct UzuChatTokenStream<B: Backend> {
    // Drop order matters (fields drop top-to-bottom):
    //   1. `decoding_loop` borrows the guards -> must drop first
    //      (its Drop also writes `state.next_seed_token` back, while locks are still held)
    //   2. the guards release the locks
    //   3. the Arc<Mutex<_>> owners are released last
    decoding_loop: Box<dyn Iterator<Item = Result<u64, LanguageModelStreamError<B>>>>,
    state_guard: MutexGuard<'static, LanguageModelState<B>>,
    model_guard: MutexGuard<'static, LanguageModel<B>>,
    _state: Container<LanguageModelState<B>>,
    _model: Container<LanguageModel<B>>,
    finished: bool,
}

impl<B: Backend> UzuChatTokenStream<B> {
    pub fn new(
        state: Container<LanguageModelState<B>>,
        model: Container<LanguageModel<B>>,
        input: &Vec<u64>,
        config: ChatReplyConfig,
    ) -> Result<Self, Error> {
        // SAFETY: the guards are extended to `'static`, which is sound because the
        // `Arc<Mutex<_>>` allocations they lock are owned by `_model`/`_state` in the same
        // struct, and the guards (and the iterator borrowing them) are dropped before those
        // owners. The Mutex value lives behind the Arc, so moving the guards does not move it.
        let model_guard: MutexGuard<'static, LanguageModel<B>> = match model.value.lock() {
            Ok(guard) => unsafe { std::mem::transmute(guard) },
            Err(err) => return Err(Error::from(err.to_string())),
        };
        let mut state_guard: MutexGuard<'static, LanguageModelState<B>> = match state.value.lock() {
            Ok(guard) => unsafe { std::mem::transmute(guard) },
            Err(err) => return Err(Error::from(err.to_string())),
        };

        // SAFETY: the referenced data outlives the iterator (see field docs on
        // `UzuChatTokenStream`). The guards are only kept to hold the locks and are never
        // touched again, so the iterator is the sole active borrower.
        let model_ref: &'static LanguageModel<B> = unsafe { &*(&*model_guard as *const LanguageModel<B>) };
        let state_ref: &'static mut LanguageModelState<B> =
            unsafe { &mut *(&mut *state_guard as *mut LanguageModelState<B>) };

        let options = Self::get_stream_options(model_ref, config);
        let decoding_loop = match model_ref.stream(&input, state_ref, options) {
            Ok(stream) => Box::new(stream) as Box<dyn Iterator<Item = Result<u64, LanguageModelStreamError<B>>>>,
            Err(err) => return Err(Error::from(err.to_string())),
        };

        Ok(Self {
            decoding_loop,
            state_guard,
            model_guard,
            _state: state,
            _model: model,
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

        if let Some(max_tokens) = self.model_guard.recommended_context_length()
            && self.state_guard.tokens().len() >= max_tokens
        {
            self.finished = true;
            return Poll::Ready(None);
        }

        let option_result = self.decoding_loop.next().map(|res| {
            let result = res.map_err(|err| {
                self.finished = true;
                Error::from(err.to_string())
            });
            result.iter().for_each(|token| {
                if self.model_guard.generation_config().stop_token_ids.contains(&token) {
                    self.finished = true;
                }
            });
            result
        });

        Poll::Ready(option_result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.decoding_loop.size_hint()
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
