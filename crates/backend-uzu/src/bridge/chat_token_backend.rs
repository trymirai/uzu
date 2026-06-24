use std::{path::PathBuf, pin::Pin};

use futures::{Stream, StreamExt, TryStreamExt, stream};
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
use tokio_util::sync::CancellationToken;

use crate::{
    backends::common::Backend,
    bridge::{
        chat_token_state::UzuChatTokenBackendInstanceState,
        helpers::{get_grammar, get_max_context_length, get_sampling_method, get_speculator},
        sync_shared::SyncShared,
    },
    engine::{
        Engine,
        language_model::{
            LanguageModel,
            stream::{LanguageModelStreamError, LanguageModelStreamOptions},
        },
    },
};

pub struct UzuChatTokenBackendInstance<B: Backend> {
    model: SyncShared<LanguageModel<B>>,
    config: ChatConfig,
}

impl<B: Backend> UzuChatTokenBackendInstance<B> {
    pub fn new(
        model_path: String,
        config: ChatConfig,
    ) -> Result<Self, BackendError> {
        let engine = Engine::<B>::new().map_err(|err| err.to_string())?;
        let model_path = PathBuf::from(model_path);
        let model = engine.load_language_model(&model_path).map_err(|err| err.to_string())?;
        Ok(Self {
            model: SyncShared::new(model),
            config,
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

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        state: &'a mut dyn State,
        config: Self::StreamConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, BackendError>> + Send + 'a>> {
        let mut state_guard = match state.as_any_mut().downcast_mut::<UzuChatTokenBackendInstanceState<B>>() {
            Some(state) => match state.value.lock() {
                Ok(state) => state,
                Err(err) => return error_stream(err.to_string()),
            },
            None => return error_stream("unexpected state type for uzu chat token instance".to_string()),
        };
        let model_guard = match self.model.lock() {
            Ok(model) => model,
            Err(err) => return error_stream(err.to_string()),
        };

        let iterator_options = LanguageModelStreamOptions {
            sampling_method: get_sampling_method::<B>(&model_guard, &config.sampling_policy),
            grammar: get_grammar::<B>(config.grammar),
            speculator: get_speculator(self.config.speculation_preset.clone()),
        };
        let iterator: Box<dyn Iterator<Item = Result<u64, LanguageModelStreamError<B>>>> =
            match model_guard.stream(input, &mut state_guard, iterator_options) {
                Ok(iter) => match config.token_limit {
                    Some(limit) => Box::new(iter.take(limit as usize)),
                    None => Box::new(iter),
                },
                Err(err) => return error_stream(err.to_string()),
            };

        // TODO agolokoz: replace with async streaming
        let mut tokens = Vec::<u64>::new();
        for result in iterator {
            match result {
                Ok(token) => tokens.push(token),
                Err(err) => {
                    return error_stream(err.to_string());
                },
            }
        }
        let stream = stream::iter(tokens).map(|token| Result::<u64, BackendError>::Ok(token));
        Box::pin(stream)
    }
}

fn error_stream<'a>(
    message: String
) -> Pin<Box<dyn Stream<Item = Result<ChatTokenStreamOutput, BackendError>> + Send + 'a>> {
    Box::pin(stream::once(async move {
        Err::<ChatTokenStreamOutput, BackendError>(Box::<dyn std::error::Error + Send + Sync>::from(message))
    }))
}
