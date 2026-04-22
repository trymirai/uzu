mod config;

use std::{future::Future, pin::Pin};

use async_openai::{Client, config::OpenAIConfig};
pub use config::Config;
use fancy_regex::Regex;
use serde::Deserialize;
use shoji::{
    traits::Registry as RegistryTrait,
    types::model::{Model as ShojiModel, ModelAccessibility, ModelEntity, ModelEntityType, ModelSpecialization},
};

#[derive(Debug, Deserialize)]
struct Model {
    id: String,
}

#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    data: Vec<Model>,
}

use crate::registry::Error;

pub struct Registry {
    config: Config,
    client: Client<OpenAIConfig>,
    model_filter: Option<Regex>,
}

impl Registry {
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

        let client = Client::with_config(openai_config);

        let model_filter = config.model_filter_pattern.as_deref().map(Regex::new).transpose().map_err(|error| {
            Error::UnableToCreate {
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
    type Error = Error;

    fn indentifier(&self) -> String {
        self.config.identifier.clone()
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<ShojiModel>, Error>> + Send + '_>> {
        Box::pin(async {
            let response: ListModelsResponse =
                self.client.models().list_byot().await.map_err(|error| Error::UnableToGetModels {
                    message: error.to_string(),
                })?;
            let mut identifiers = response.data.into_iter().map(|model| model.id).collect::<Vec<_>>();
            identifiers.sort();

            let registry_entity = self.create_entity(ModelEntityType::Registry);
            let backend_entity = self.create_entity(ModelEntityType::Backend);
            let vendor_entity = self.create_entity(ModelEntityType::Vendor);
            let family_entity = self.create_entity(ModelEntityType::Family);

            let models = identifiers
                .into_iter()
                .filter(|identifier| {
                    self.model_filter.as_ref().is_none_or(|regex| regex.is_match(identifier).unwrap_or(false))
                })
                .map(|identifier| {
                    let variant_entity = ModelEntity {
                        r#type: ModelEntityType::Variant,
                        identifier: identifier.clone(),
                        parent_identifier: None,
                        name: identifier.clone(),
                        description: None,
                        version: None,
                        icons: vec![],
                    };
                    let model = ShojiModel {
                        identifier: identifier.clone(),
                        entities: vec![
                            registry_entity.clone(),
                            backend_entity.clone(),
                            vendor_entity.clone(),
                            family_entity.clone(),
                            variant_entity,
                        ],
                        specializations: vec![ModelSpecialization::Chat],
                        number_of_parameters: None,
                        quantization: None,
                        accessibility: ModelAccessibility::Remote {
                            repository: None,
                        },
                    };
                    model
                })
                .collect();

            Ok(models)
        })
    }
}

impl Registry {
    fn create_entity(
        &self,
        r#type: ModelEntityType,
    ) -> ModelEntity {
        ModelEntity {
            r#type,
            identifier: self.config.identifier.clone(),
            parent_identifier: None,
            name: self.config.name.clone(),
            description: None,
            version: None,
            icons: vec![],
        }
    }
}
