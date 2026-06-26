use std::pin::Pin;

use shoji::{
    traits::{
        Backend as LlmBackend,
        backend::{
            Error as BackendError,
            chat_token::{Backend as ChatTokenBackend, Instance as ChatTokenInstance},
            classification::{Backend as ClassificationBackend, Config, Instance as ClassificationInstance},
        },
    },
    types::session::chat::ChatConfig,
};
use tokenizers::Tokenizer;

use crate::{
    TOOLCHAIN_VERSION,
    backends::select_backend,
    bridge::{
        chat_token_backend::UzuChatTokenBackendInstance, classification_backend::UzuClassificationBackendInstance,
    },
};

pub struct UzuLlmBackend;

impl UzuLlmBackend {
    pub fn new() -> Self {
        Self {}
    }
}

impl LlmBackend for UzuLlmBackend {
    fn identifier(&self) -> String {
        "uzu".to_string()
    }

    fn version(&self) -> String {
        TOOLCHAIN_VERSION.to_string()
    }

    fn as_chat_via_token_capable(&self) -> Option<&dyn ChatTokenBackend> {
        Some(self)
    }

    fn as_classification_capable(&self) -> Option<&dyn ClassificationBackend> {
        Some(self)
    }
}

impl ChatTokenBackend for UzuLlmBackend {
    fn instance<'a>(
        &'a self,
        reference: String,
        config: ChatConfig,
        tokenizer: &'a Tokenizer,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn ChatTokenInstance>, BackendError>> + Send + 'a>> {
        Box::pin(async move {
            let instance = select_backend!(
                UzuChatTokenBackendInstance::<B>::new(reference, config, tokenizer)
                    .map(|i| Box::new(i) as Box<dyn ChatTokenInstance>),
                BackendError::from("Unable to open any backend")
            )?;
            Ok(instance)
        })
    }
}

impl ClassificationBackend for UzuLlmBackend {
    fn instance(
        &self,
        reference: String,
        _config: Config,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn ClassificationInstance>, BackendError>> + Send + '_>> {
        Box::pin(async move {
            let instance = select_backend!(
                UzuClassificationBackendInstance::<B>::new(reference)
                    .map(|i| Box::new(i) as Box<dyn ClassificationInstance>),
                BackendError::from("Unable to open any backend")
            )?;
            Ok(instance)
        })
    }
}
