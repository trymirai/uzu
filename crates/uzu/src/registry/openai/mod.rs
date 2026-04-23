mod config;

use std::{future::Future, pin::Pin};

use async_openai::{Client, config::OpenAIConfig};
pub use config::Config;
use fancy_regex::Regex;
use serde::Deserialize;
use shoji::{
    traits::Registry as RegistryTrait,
    types::model::{Model as ShojiModel, ModelAccessibility, ModelSpecialization},
};

#[derive(Debug, Deserialize)]
struct Model {
    id: String,
}

#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    data: Vec<Model>,
}

use crate::registry::RegistryError;

pub struct Registry {
    config: Config,
    client: Client<OpenAIConfig>,
    model_filter: Option<Regex>,
}

impl Registry {
    pub fn new(config: Config) -> Result<Self, RegistryError> {
        let mut openai_config = OpenAIConfig::new().with_api_base(&config.api_endpoint);
        if let Some(api_key) = config.api_key.as_ref() {
            openai_config = openai_config.with_api_key(api_key);
        }
        if let Some(headers) = config.headers.as_ref() {
            for (key, value) in headers {
                let name = reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|error| {
                    RegistryError::UnableToCreate {
                        message: error.to_string(),
                    }
                })?;
                openai_config =
                    openai_config.with_header(name, value.as_str()).map_err(|error| RegistryError::UnableToCreate {
                        message: error.to_string(),
                    })?;
            }
        }

        let client = Client::with_config(openai_config);

        let model_filter = config.model_filter_pattern.as_deref().map(Regex::new).transpose().map_err(|error| {
            RegistryError::UnableToCreate {
                message: error.to_string(),
            }
        })?;

        Ok(Self {
            config,
            client,
            model_filter,
        })
    }
}

impl RegistryTrait for Registry {
    type Error = RegistryError;

    fn indentifier(&self) -> String {
        self.config.identifier.clone()
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<ShojiModel>, RegistryError>> + Send + '_>> {
        Box::pin(async {
            let response: ListModelsResponse =
                self.client.models().list_byot().await.map_err(|error| RegistryError::UnableToGetModels {
                    message: error.to_string(),
                })?;
            let mut identifiers = response.data.into_iter().map(|model| model.id).collect::<Vec<_>>();
            identifiers.sort();

            let models = identifiers
                .into_iter()
                .filter(|identifier| {
                    self.model_filter.as_ref().is_none_or(|regex| regex.is_match(identifier).unwrap_or(false))
                })
                .map(|identifier| {
                    let model = ShojiModel::external(
                        identifier.clone(),
                        self.config.identifier.clone(),
                        self.config.name.clone(),
                        self.config.identifier.clone(),
                        self.config.name.clone(),
                        "default".to_string(),
                        vec![ModelSpecialization::Chat],
                        ModelAccessibility::Remote {
                            repository: None,
                        },
                    );
                    model
                })
                .collect();

            Ok(models)
        })
    }
}
