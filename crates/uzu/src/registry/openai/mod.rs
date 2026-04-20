mod config;

use std::{future::Future, pin::Pin};

pub use config::Config;
use fancy_regex::Regex;
use openai_api_rs::v1::api::OpenAIClient;
use shoji::{
    traits::Registry as RegistryTrait,
    types::model::{Accessibility, Entity, EntityType, Model, Specialization},
};

use crate::registry::Error;

pub struct Registry {
    config: Config,
    client: OpenAIClient,
    model_filter: Option<Regex>,
}

impl Registry {
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

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, Error>> + Send + '_>> {
        Box::pin(async {
            let response = self.client.list_models().await.map_err(|error| Error::UnableToGetModels {
                message: error.to_string(),
            })?;
            let mut identifiers =
                response.inner.data.into_iter().filter_map(|model| model.id.clone()).collect::<Vec<_>>();
            identifiers.sort();

            let registry_entity = self.create_entity(EntityType::Registry);
            let backend_entity = self.create_entity(EntityType::Backend);
            let vendor_entity = self.create_entity(EntityType::Vendor);
            let family_entity = self.create_entity(EntityType::Family);

            let models = identifiers
                .into_iter()
                .filter(|identifier| {
                    self.model_filter.as_ref().is_none_or(|regex| regex.is_match(identifier).unwrap_or(false))
                })
                .map(|identifier| {
                    let variant_entity = Entity {
                        r#type: EntityType::Variant,
                        identifier: identifier.clone(),
                        parent_identifier: None,
                        name: identifier.clone(),
                        description: None,
                        version: None,
                        icons: vec![],
                    };
                    let model = Model {
                        identifier: identifier.clone(),
                        entities: vec![
                            registry_entity.clone(),
                            backend_entity.clone(),
                            vendor_entity.clone(),
                            family_entity.clone(),
                            variant_entity,
                        ],
                        specializations: vec![Specialization::Chat],
                        number_of_parameters: None,
                        quantization: None,
                        accessibility: Accessibility::Remote {
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
        r#type: EntityType,
    ) -> Entity {
        Entity {
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
