mod api;
mod config;
mod types;

use std::{
    fs::{read_to_string, write},
    future::Future,
    path::PathBuf,
    pin::Pin,
    time::Duration,
};

pub use api::Endpoint;
pub use config::{Backend, Config};
use indexmap::IndexMap;
use nagare::api::{Client, Config as ClientConfig, Error as ApiError};
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
            match self.fetch_models().await {
                Ok(models) => {
                    if let Err(error) = self.save_registry(&models) {
                        tracing::warn!(?error, "failed to save Mirai registry");
                    }
                    Ok(models)
                },
                Err(error) => {
                    let transient = matches!(error, ApiError::Timeout | ApiError::Network(_))
                        || matches!(error, ApiError::Http { code, .. } if code >= 500);
                    if transient && let Ok(models) = self.load_registry() {
                        tracing::warn!(?error, "serving cached Mirai registry after fetch failure");
                        return Ok(models);
                    }
                    Err(RegistryError::UnableToGetModels {
                        message: error.to_string(),
                    })
                },
            }
        })
    }
}

impl Registry {
    async fn fetch_models(&self) -> Result<Vec<Model>, ApiError> {
        let response: Response = self
            .client
            .response(&Endpoint::FetchModels {
                device: self.config.device.clone(),
                backends: self.config.backends.clone(),
                include_traces: self.config.include_traces,
            })
            .await?;
        response.models().ok_or_else(|| ApiError::Decode("response contained no models".to_string()))
    }

    fn registry_path(&self) -> PathBuf {
        self.config.cache_path.join("registry.json")
    }

    fn save_registry(
        &self,
        models: &[Model],
    ) -> Result<(), RegistryError> {
        let contents = serde_json::to_vec_pretty(models).map_err(|error| RegistryError::UnableToGetModels {
            message: format!("Unable to serialize registry: {}", error),
        })?;
        write(self.registry_path(), contents).map_err(|error| RegistryError::UnableToGetModels {
            message: format!("Unable to write registry: {}", error),
        })
    }

    fn load_registry(&self) -> Result<Vec<Model>, RegistryError> {
        let contents = read_to_string(self.registry_path()).map_err(|error| RegistryError::UnableToGetModels {
            message: format!("Unable to read registry: {}", error),
        })?;
        serde_json::from_str(&contents).map_err(|error| RegistryError::UnableToGetModels {
            message: format!("Unable to parse registry: {}", error),
        })
    }
}
