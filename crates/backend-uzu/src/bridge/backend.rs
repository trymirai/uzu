use std::pin::Pin;

use shoji::{
    traits::{
        Backend as LlmBackend,
        backend::{
            Error as BackendError,
            chat_token::{Backend as ChatTokenBackend, Instance as ChatTokenInstance},
        },
    },
    types::session::chat::ChatConfig,
};
use tokenizers::Tokenizer;

use crate::{TOOLCHAIN_VERSION, backends::select_backend, bridge::chat_token_backend::UzuChatTokenBackendInstance};

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
