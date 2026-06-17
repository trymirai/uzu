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
use nagare::api::{Client, Config as ClientConfig};
use reqwest::header::AUTHORIZATION;
use serde::{Deserialize, Serialize};
use shoji::{traits::Registry as RegistryTrait, types::model::Model};
pub use types::Response;
use uuid::Uuid;

use crate::registry::RegistryError;

#[derive(Serialize, Deserialize)]
struct RegistryCache {
    key: String,
    models: Vec<Model>,
}

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
                Err(fetch_error) => match self.load_registry() {
                    Ok(models) => {
                        tracing::warn!(?fetch_error, "serving cached Mirai registry after fetch failure");
                        Ok(models)
                    },
                    Err(load_error) => {
                        tracing::warn!(?load_error, "failed to load cached Mirai registry");
                        Err(fetch_error)
                    },
                },
            }
        })
    }
}

impl Registry {
    async fn fetch_models(&self) -> Result<Vec<Model>, RegistryError> {
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
        response.models().ok_or(RegistryError::UnableToGetModels {
            message: "Unable to extract from response".to_string(),
        })
    }

    fn registry_path(&self) -> PathBuf {
        self.config.cache_path.join("registry.json")
    }

    fn cache_key(&self) -> String {
        let fingerprint = serde_json::json!({
            "api_key": self.config.api_key,
            "backends": self.config.backends,
            "include_traces": self.config.include_traces,
            "os_name": self.config.device.os_name,
            "cpu_name": self.config.device.cpu_name,
            "memory_total": self.config.device.memory_total,
        });
        let bytes = serde_json::to_vec(&fingerprint).unwrap_or_default();
        Uuid::new_v5(&Uuid::NAMESPACE_URL, &bytes).simple().to_string()
    }

    fn save_registry(
        &self,
        models: &[Model],
    ) -> Result<(), RegistryError> {
        let cache = RegistryCache {
            key: self.cache_key(),
            models: models.to_vec(),
        };
        let contents = serde_json::to_vec_pretty(&cache).map_err(|error| RegistryError::UnableToGetModels {
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
        let cache: RegistryCache =
            serde_json::from_str(&contents).map_err(|error| RegistryError::UnableToGetModels {
                message: format!("Unable to parse registry: {}", error),
            })?;
        if cache.key != self.cache_key() {
            return Err(RegistryError::UnableToGetModels {
                message: "Cached registry does not match the current request context".to_string(),
            });
        }
        Ok(cache.models)
    }
}
