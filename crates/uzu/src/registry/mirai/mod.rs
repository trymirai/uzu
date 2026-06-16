mod api;
mod config;
mod snapshot;
mod types;

use std::{collections::HashMap, future::Future, pin::Pin, time::Duration};

pub use api::Endpoint;
pub use config::{Backend, Config};
use indexmap::IndexMap;
use nagare::api::{Client, Config as ClientConfig};
use reqwest::header::AUTHORIZATION;
use shoji::{traits::Registry as RegistryTrait, types::model::Model};
use snapshot::{load_snapshot, save_snapshot, scan_model_metadata};
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
            match self.fetch_models().await {
                Ok(response) => {
                    if let Err(error) = save_snapshot(&self.config.cache_path, &response) {
                        tracing::warn!(?error, "failed to save Mirai registry snapshot");
                    }
                    response.models().ok_or(RegistryError::UnableToGetModels {
                        message: "Unable to extract from response".to_string(),
                    })
                },
                Err(fetch_error) => {
                    let mut models_by_identifier = HashMap::new();

                    match load_snapshot(&self.config.cache_path) {
                        Ok(response) => match response.models() {
                            Some(models) => {
                                for model in models {
                                    models_by_identifier.insert(model.identifier.clone(), model);
                                }
                            },
                            None => {
                                tracing::warn!("failed to extract models from cached Mirai registry snapshot");
                            },
                        },
                        Err(error) => {
                            tracing::warn!(?error, "failed to load Mirai registry snapshot");
                        },
                    }

                    for model in scan_model_metadata(&self.config.cache_path) {
                        models_by_identifier.entry(model.identifier.clone()).or_insert(model);
                    }

                    if models_by_identifier.is_empty() {
                        Err(fetch_error)
                    } else {
                        Ok(models_by_identifier.into_values().collect())
                    }
                },
            }
        })
    }
}

impl Registry {
    async fn fetch_models(&self) -> Result<Response, RegistryError> {
        self.client
            .response(&Endpoint::FetchModels {
                device: self.config.device.clone(),
                backends: self.config.backends.clone(),
                include_traces: self.config.include_traces,
            })
            .await
            .map_err(|error| RegistryError::UnableToGetModels {
                message: error.to_string(),
            })
    }
}
