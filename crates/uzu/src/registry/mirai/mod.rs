mod api;
mod config;
mod types;

use std::{future::Future, pin::Pin, time::Duration};

pub use api::Endpoint;
pub use config::{Backend, Config};
use indexmap::IndexMap;
use nagare::api::{Client, Config as ClientConfig};
use reqwest::header::AUTHORIZATION;
use shoji::{traits::Registry as RegistryTrait, types::model::Model};
pub use types::Response;

use crate::registry::RegistryError;

pub struct Registry {
    config: Config,
    client: Client,
}

impl Registry {
    pub fn new(config: Config) -> Result<Self, RegistryError> {
        let mut headers: IndexMap<String, String> = IndexMap::new();
        if let Some(api_key) = config.api_key.clone() {
            headers.insert(AUTHORIZATION.to_string(), format!("Bearer {}", api_key));
        }

        let client_config =
            ClientConfig::new("https://sdk.trymirai.com/api/v1".to_string(), Duration::from_secs(10), headers);
        let client = Client::new(client_config).map_err(|error| RegistryError::UnableToCreate {
            message: error.to_string(),
        })?;

        Ok(Self {
            config,
            client,
        })
    }
}
impl RegistryTrait for Registry {
    type Error = RegistryError;

    fn indentifier(&self) -> String {
        "mirai".to_string()
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, RegistryError>> + Send + '_>> {
        Box::pin(async {
            let response: Response = self
                .client
                .response(&Endpoint::FetchModels {
                    device: self.config.device.clone(),
                    backends: self.config.backends.clone(),
                    include_traces: self.config.include_traces,
                })
                .await
                .map_err(|error| RegistryError::UnableToGetModels {
                    message: error.to_string(),
                })?;
            let models = response.models().ok_or(RegistryError::UnableToGetModels {
                message: "Unable to extract from response".to_string(),
            })?;
            Ok(models)
        })
    }
}
