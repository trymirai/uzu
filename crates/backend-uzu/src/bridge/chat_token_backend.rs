use std::{
    any::Any,
    path::PathBuf,
    pin::Pin,
    task::{Context, Poll},
};

use futures::{Stream, StreamExt};
use shoji::{
    traits::{
        State,
        backend::{
            Error as BackendError, Instance as BackendInstance,
            chat_token::{
                Instance as ChatTokenBackendInstance, StreamInput as ChatTokenStreamInput,
                StreamOutput as ChatTokenStreamOutput, TokenStreamOutput,
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
        sync_shared::SyncShared,
    },
    engine::{
        Engine,
        language_model::{
            LanguageModel,
            stream::{LanguageModelStreamOptions, LanguageModelStreamSpeculatorOptions},
        },
    },
    speculators::speculator::Speculator,
};

pub struct UzuChatTokenBackendInstance<B: Backend> {
    engine: SyncShared<Engine<B>>,
    model: SyncShared<LanguageModel<B>>,
    config: ChatConfig,
    tokenizer: Tokenizer,
    stop_token_ids: Vec<i32>,
    speculator: Option<(Box<dyn Speculator>, usize)>,
    max_context_length: Option<usize>,
}

impl<B: Backend> UzuChatTokenBackendInstance<B> {
    pub fn new(
        model_path: String,
        config: ChatConfig,
        tokenizer: &Tokenizer,
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
            engine: SyncShared::new(engine),
            model: SyncShared::new(model),
            config,
            tokenizer: tokenizer.clone(),
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

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn State>, BackendError>> + Send + '_>> {
        Box::pin(async move {
            let model = self.model.lock().map_err(|err| BackendError::from(err.to_string()))?;
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
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, BackendError>> + Send + 'a>> {
        let model = self.model.clone();
        let state = match (state as &mut dyn Any).downcast_mut::<UzuChatTokenBackendInstanceState<B>>() {
            Some(state) => state.value.clone(),
            None => return error_stream("unexpected state type for uzu chat token instance".to_string()),
        };
        let token_limit = config.token_limit.map(|count| count as usize).unwrap_or(usize::MAX);

        #[allow(clippy::await_holding_lock)]
        let stream = async_stream::stream! {
            let model_guard = match model.lock() {
                Ok(model) => model,
                Err(err) => {
                    yield Err(BackendError::from(err.to_string()));
                    return;
                },
            };
            let mut state_guard = match state.lock() {
                Ok(state) => state,
                Err(err) => {
                    yield Err(BackendError::from(err.to_string()));
                    return;
                },
            };

            let grammar = if let Some(grammar_config) = config.grammar {
                match get_grammar(grammar_config, &self.tokenizer, &self.stop_token_ids) {
                    Ok(grammar) => Some(grammar),
                    Err(err) => {
                        yield Err(BackendError::from(err.to_string()));
                        return
                    }
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

            yield Ok(TokenStreamOutput::PrefillStarted);
            let iterator = match model_guard.stream(input, &mut state_guard, options) {
                Ok(iter) => iter,
                Err(err) => {
                    yield Err(BackendError::from(err.to_string()));
                    return;
                },
            };

            // here additional condition to prevent iterator start
            if token_limit == 0 {
                yield Ok(TokenStreamOutput::LimitReached);
                return;
            }

            let mut token_count = 0usize;
            for result in iterator.take(token_limit) {
                match result {
                    Ok(token) => {
                        yield Ok(TokenStreamOutput::Token(token));
                        token_count += 1;
                    },
                    Err(err) => {
                        yield Err(BackendError::from(err.to_string()));
                        return;
                    },
                }
            }
            if token_count >= token_limit {
                yield Ok(TokenStreamOutput::LimitReached);
            }
        };

        Box::pin(AssertSend(stream).take_until(cancel_token.cancelled_owned()))
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        self.engine.lock().ok().and_then(|engine| engine.peak_memory_usage())
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

struct AssertSend<S>(S);

unsafe impl<S> Send for AssertSend<S> {}

impl<S: Stream> Stream for AssertSend<S> {
    type Item = S::Item;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        // SAFETY: we never move `self.0` out of the pinned reference.
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.0) };
        inner.poll_next(cx)
    }
}
