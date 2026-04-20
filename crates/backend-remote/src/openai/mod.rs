mod config;
mod error;
mod instance;

use std::{pin::Pin, sync::Arc};

pub use config::Config;
pub use error::Error;
pub use instance::Instance;
use openai_api_rs::v1::api::OpenAIClient;
use shoji::traits::{
    Backend as BackendTrait,
    backend::{
        Error as BackendError,
        chat_message::{self},
    },
};

pub struct Backend {
    config: Config,
    client: Arc<OpenAIClient>,
}

impl Backend {
    pub fn new(config: Config) -> Result<Self, Error> {
        let mut client_builder = OpenAIClient::builder().with_endpoint(config.api_endpoint.clone());
        if let Some(api_key) = config.api_key.clone() {
            client_builder = client_builder.with_api_key(api_key);
        }
        if let Some(headers) = config.headers.clone() {
            for (key, value) in headers.iter() {
                client_builder = client_builder.with_header(key, value);
            }
        }

        let client = client_builder.build().map_err(|error| Error::UnableToCreate {
            message: error.to_string(),
        })?;

        Ok(Self {
            config,
            client: Arc::new(client),
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
        _config: chat_message::Config,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn chat_message::Instance>, BackendError>> + Send + '_>> {
        let client = self.client.clone();
        Box::pin(async move { Ok(Box::new(Instance::new(client, reference)) as Box<dyn chat_message::Instance>) })
    }
}
