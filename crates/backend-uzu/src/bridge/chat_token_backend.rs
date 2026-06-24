use std::{path::PathBuf, pin::Pin};

use futures::{Stream, StreamExt, stream};
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

        let mut grammar_opt = if let Some(grammar_config) = config.grammar {
            match get_grammar::<B>(grammar_config, &self.tokenizer, &self.stop_token_ids) {
                Ok(grammar) => Some(grammar),
                Err(err) => return error_stream(err.to_string()),
            }
        } else {
            None
        };

        let spec_options = if let Some(ref speculator) = self.speculator {
            Some(LanguageModelStreamSpeculatorOptions {
                speculator: speculator.as_ref(),
                speculation_budget: 0,
                trie_creation_config: Default::default(),
            })
        } else {
            None
        };

        let iterator_options = LanguageModelStreamOptions {
            sampling_method: get_sampling_method::<B>(&model_guard, &config.sampling_policy),
            grammar: grammar_opt.as_deref_mut(),
            speculator: spec_options,
        };

        let iterator = match model_guard.stream(input, &mut state_guard, iterator_options) {
            Ok(iter) => iter,
            Err(err) => return error_stream(err.to_string()),
        };

        // // TODO agolokoz: replace with async streaming
        let mut tokens = Vec::<u64>::new();
        for result in iterator {
            match result {
                Ok(token) => {
                    // TODO agolokoz: move stop token ids checking to nagare
                    tokens.push(token);
                    if model_guard.generation_config().stop_token_ids.contains(&token) {
                        break;
                    }
                },
                Err(err) => {
                    return error_stream(err.to_string());
                },
            }
        }

        // box pin?
        // struct {
        // guard
        // guard
        // stream
        // }
        // transmute

        let stream = stream::iter(tokens)
            .map(|token| Result::<u64, BackendError>::Ok(token))
            .take_until(cancel_token.cancelled_owned());
        Box::pin(stream)
    }
}

// struct UzuTokenStream<B> {
//     model_guard: Pin<Box<MutexGuard<'static, LanguageModel<B>>>>,
//     state_guard: Pin<Box<MutexGuard<'static, LanguageModelState<B>>>>,
//     iterator: Pin<Box<dyn Stream<Item = Result<ChatTokenStreamOutput, BackendError>> + Send + Sync + 'static>>,
// }
//
// impl<B: Backend> UzuTokenStream<B> {
//     pub fn new(
//         model_guard: MutexGuard<'static, LanguageModel<B>>,
//         state_guard: MutexGuard<'static, LanguageModelState<B>>,
//         iterator: Pin<Box<dyn Stream<Item = Result<ChatTokenStreamOutput, BackendError>> + Send + Sync + 'static>>,
//     ) -> Pin<Box<Self>> {
//         // todo: transumte + add comment
//         Box::pin(Self {
//             model_guard: Box::pin(model_guard),
//             state_guard: Box::pin(state_guard),
//             iterator,
//         })
//     }
// }
