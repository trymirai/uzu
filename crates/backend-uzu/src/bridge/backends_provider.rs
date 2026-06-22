use std::pin::Pin;

use shoji::{
    traits::{
        Backend as LlmBackend,
        backend::{
            Error,
            chat_token::{Backend as ChatTokenLlmInstanceProvider, Instance as ChatTokenLlmInstance},
        },
    },
    types::session::chat::ChatConfig,
};

use crate::{TOOLCHAIN_VERSION, backends::select_backend, bridge::chat_token::UzuChatTokenLlmInstance};

pub struct UzuLlmBackend;

impl UzuLlmBackend {
    pub fn new() -> Self {
        Self
    }
}

impl LlmBackend for UzuLlmBackend {
    fn identifier(&self) -> String {
        "uzu".to_string()
    }

    fn version(&self) -> String {
        TOOLCHAIN_VERSION.to_string()
    }

    fn as_chat_via_token_capable(&self) -> Option<&dyn ChatTokenLlmInstanceProvider> {
        Some(self)
    }
}

impl ChatTokenLlmInstanceProvider for UzuLlmBackend {
    fn instance(
        &self,
        reference: String,
        _config: ChatConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn ChatTokenLlmInstance>, Error>> + Send + '_>> {
        Box::pin(async move {
            let instance = select_backend!(
                UzuChatTokenLlmInstance::<B>::new(reference).map(|i| Box::new(i) as Box<dyn ChatTokenLlmInstance>),
                Error::from("Unable to open any backend")
            )?;
            Ok(instance)
        })
    }
}
