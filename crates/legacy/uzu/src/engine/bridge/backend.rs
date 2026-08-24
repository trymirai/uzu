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
use uzu_engine::{
    TOOLCHAIN_VERSION,
    backends::{BackendSelection, common::Backend, select_backend},
};

use crate::engine::bridge::{
    chat_token_backend::UzuChatTokenBackendInstance, classification_backend::UzuClassificationBackendInstance,
};

struct ChatTokenInstanceSelection {
    reference: String,
    config: ChatConfig,
}

impl BackendSelection for ChatTokenInstanceSelection {
    type Output = Box<dyn ChatTokenInstance>;
    type Error = BackendError;

    fn select<B: Backend>(self) -> Result<Self::Output, Self::Error> {
        UzuChatTokenBackendInstance::<B>::new(self.reference, self.config)
            .map(|instance| Box::new(instance) as Box<dyn ChatTokenInstance>)
    }
}

struct ClassificationInstanceSelection {
    reference: String,
}

impl BackendSelection for ClassificationInstanceSelection {
    type Output = Box<dyn ClassificationInstance>;
    type Error = BackendError;

    fn select<B: Backend>(self) -> Result<Self::Output, Self::Error> {
        UzuClassificationBackendInstance::<B>::new(self.reference)
            .map(|instance| Box::new(instance) as Box<dyn ClassificationInstance>)
    }
}

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
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn ChatTokenInstance>, BackendError>> + Send + 'a>> {
        Box::pin(async move {
            let instance = select_backend(
                ChatTokenInstanceSelection {
                    reference,
                    config,
                },
                BackendError::from("Unable to open any backend"),
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
            let instance = select_backend(
                ClassificationInstanceSelection {
                    reference,
                },
                BackendError::from("Unable to open any backend"),
            )?;
            Ok(instance)
        })
    }
}
