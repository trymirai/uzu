mod bridging;
mod config;
mod error;
mod instance;
mod stream_state;
mod tool_call_state;

use std::{pin::Pin, sync::Arc};

use async_openai::{Client, config::OpenAIConfig};
pub use config::{ApiType, Config};
pub use error::Error;
pub use instance::Instance;
use shoji::{
    traits::{
        Backend as BackendTrait,
        backend::{Error as BackendError, chat_message},
    },
    types::session::chat::Config as ChatConfig,
};

pub struct Backend {
    config: Config,
    client: Arc<Client<OpenAIConfig>>,
}

impl Backend {
    pub fn new(config: Config) -> Result<Self, Error> {
        let mut openai_config = OpenAIConfig::new().with_api_base(&config.api_endpoint);
        if let Some(api_key) = config.api_key.as_ref() {
            openai_config = openai_config.with_api_key(api_key);
        }
        if let Some(headers) = config.headers.as_ref() {
            for (key, value) in headers {
                let name =
                    reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|error| Error::UnableToCreate {
                        message: error.to_string(),
                    })?;
                openai_config =
                    openai_config.with_header(name, value.as_str()).map_err(|error| Error::UnableToCreate {
                        message: error.to_string(),
                    })?;
            }
        }

        Ok(Self {
            config,
            client: Arc::new(Client::with_config(openai_config)),
        })
    }
}

impl BackendTrait for Backend {
    fn identifier(&self) -> String {
        self.config.identifier.clone()
    }

    fn version(&self) -> String {
        "default".to_string()
    }

    fn as_chat_via_message_capable(&self) -> Option<&dyn chat_message::Backend> {
        Some(self)
    }
}

impl chat_message::Backend for Backend {
    fn instance(
        &self,
        reference: String,
        _config: ChatConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn chat_message::Instance>, BackendError>> + Send + '_>> {
        let client = self.client.clone();
        let api_type = self.config.api_type.clone();
        Box::pin(
            async move { Ok(Box::new(Instance::new(client, reference, api_type)) as Box<dyn chat_message::Instance>) },
        )
    }
}
