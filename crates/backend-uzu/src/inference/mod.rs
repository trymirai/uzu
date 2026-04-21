mod chat;
mod classification;
mod container;
mod error;
mod model_metadata;

use std::pin::Pin;

pub use chat::Instance as ChatInstance;
pub use classification::Instance as ClassificationInstance;
pub use container::Container;
pub use error::Error;
pub use model_metadata::{ModelMetadata, resolve_model_metadata};
use shoji::{
    traits::{
        Backend as BackendTrait,
        backend::{Error as BackendError, chat_message as chat_message_trait, classification as classification_trait},
    },
    types::session::chat::Config as ShojiChatConfig,
};

use crate::TOOLCHAIN_VERSION;

pub struct Backend;

impl Backend {
    pub fn new() -> Self {
        Self
    }
}

impl BackendTrait for Backend {
    fn identifier(&self) -> String {
        "uzu".to_string()
    }

    fn version(&self) -> String {
        TOOLCHAIN_VERSION.to_string()
    }

    fn as_chat_via_message_capable(&self) -> Option<&dyn chat_message_trait::Backend> {
        Some(self)
    }

    fn as_classification_capable(&self) -> Option<&dyn classification_trait::Backend> {
        Some(self)
    }
}

impl chat_message_trait::Backend for Backend {
    fn instance(
        &self,
        reference: String,
        config: ShojiChatConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn chat_message_trait::Instance>, BackendError>> + Send + '_>> {
        Box::pin(async move {
            let instance = ChatInstance::new(reference, config).map_err(|error| Box::new(error) as BackendError)?;
            Ok(Box::new(instance) as Box<dyn chat_message_trait::Instance>)
        })
    }
}

impl classification_trait::Backend for Backend {
    fn instance(
        &self,
        reference: String,
        _config: classification_trait::Config,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn classification_trait::Instance>, BackendError>> + Send + '_>> {
        Box::pin(async move {
            let instance = ClassificationInstance::new(reference).map_err(|error| Box::new(error) as BackendError)?;
            Ok(Box::new(instance) as Box<dyn classification_trait::Instance>)
        })
    }
}
