use std::{
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
            chat_token::{StreamInput as ChatTokenStreamInput, StreamOutput as ChatTokenStreamOutput},
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
    model: SyncShared<LanguageModel<B>>,
    config: ChatConfig,
    tokenizer: Tokenizer,
    stop_token_ids: Vec<i32>,
    speculator: Option<Box<dyn Speculator>>,
}

impl<B: Backend> UzuChatTokenBackendInstance<B> {
    pub fn new(
        model_path: String,
        config: ChatConfig,
    ) -> Result<Self, BackendError> {
        let engine = Engine::<B>::new().map_err(|err| err.to_string())?;
        let model_path = PathBuf::from(model_path);
        let model = engine.load_language_model(&model_path).map_err(|err| err.to_string())?;
        // TODO agolokoz: pass tokenizer
        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|err| BackendError::from(err.to_string()))?;
        let stop_token_ids = model.generation_config().stop_token_ids.iter().map(|id| *id as i32).collect();

        let speculator = if let Some(ref preset) = config.speculation_preset.as_ref() {
            Some(get_speculator(preset, &tokenizer).map_err(|err| BackendError::from(err))?)
        } else {
            None
        };

        Ok(Self {
            model: SyncShared::new(model),
            config,
            tokenizer,
            stop_token_ids,
            speculator,
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
                .map(|state| UzuChatTokenBackendInstanceState::new(state).clone_boxed())
        })
    }

    fn stop_token_ids(&self) -> Option<Box<[u64]>> {
        Some(self.stop_token_ids.iter().map(|id| *id as u64).collect())
    }

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        state: &'a mut dyn State,
        config: Self::StreamConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, BackendError>> + Send + 'a>> {
        let model = self.model.clone();
        let state = match state.as_any_mut().downcast_mut::<UzuChatTokenBackendInstanceState<B>>() {
            Some(state) => state.value.clone(),
            None => return error_stream("unexpected state type for uzu chat token instance".to_string()),
        };

        let stream = async_stream::stream! {
            let mut grammar = if let Some(grammar_config) = config.grammar {
                match get_grammar::<B>(grammar_config, &self.tokenizer, &self.stop_token_ids) {
                    Ok(grammar) => Some(grammar),
                    Err(err) => {
                        yield Err(BackendError::from(err.to_string()));
                        return
                    }
                }
            } else {
                None
            };
            let spec_options = self.speculator.as_ref().map(|speculator| LanguageModelStreamSpeculatorOptions {
                speculator: speculator.as_ref(),
                speculation_budget: 0,
                trie_creation_config: Default::default(),
            });

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

            let options = LanguageModelStreamOptions {
                sampling_method: get_sampling_method::<B>(&model_guard, &config.sampling_policy),
                grammar: grammar.as_deref_mut(),
                speculator: spec_options,
            };
            let iterator = match model_guard.stream(input, &mut state_guard, options) {
                Ok(iter) => iter,
                Err(err) => {
                    yield Err(BackendError::from(err.to_string()));
                    return;
                },
            };

            for result in iterator {
                match result {
                    Ok(token) => yield Ok(token),
                    Err(err) => {
                        yield Err(BackendError::from(err.to_string()));
                        return;
                    },
                }
            }
        };

        Box::pin(AssertSend(stream).take_until(cancel_token.cancelled_owned()))
    }
}

/// Wraps a stream to assert it is `Send`.
///
/// The generated decoding stream holds `MutexGuard`s (which are `!Send`) across
/// yield points, but the underlying model is driven from a single thread and the
/// shared handles already opt into `Send` via [`SyncShared`]. This wrapper
/// upholds the same contract so the stream can satisfy the `Send` bound required
/// by the backend trait.
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
