mod config;

use std::{future::Future, pin::Pin, time::Duration};

pub use config::{Backend, Config};
use indexmap::IndexMap;
use nagare::api::{Client, Config as ClientConfig, Error as ApiError};
use reqwest::header::AUTHORIZATION;
use shoji::{traits::Registry as RegistryTrait, types::model::Model};

use crate::{
    api::{FetchModelsRequest, FetchedModels},
    registry::RegistryError,
};

pub const MIRAI_API_SCHEME: &str = "https";
pub const MIRAI_API_HOST: &str = "sdk.trymirai.com";

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

        let client_config = ClientConfig::new(
            format!("{MIRAI_API_SCHEME}://{MIRAI_API_HOST}/api/v1"),
            Duration::from_secs(10),
            headers,
        );
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
            self.fetch_models().await.map_err(|error| RegistryError::UnableToGetModels {
                message: error.to_string(),
            })
        })
    }
}

impl Registry {
    async fn fetch_models(&self) -> Result<Vec<Model>, ApiError> {
        let response: FetchedModels = self
            .client
            .response(&FetchModelsRequest::new(
                self.config.device.clone(),
                self.config.backends.clone(),
                self.config.include_traces,
                std::env::var("UZU_REGISTRY_SHOW_ALL").is_ok(),
            ))
            .await?;
        response.models().map_err(ApiError::Decode)
    }
}
