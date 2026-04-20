mod config;

use std::{fs, future::Future, path::Path, pin::Pin};

pub use config::Config;
use shoji::{
    traits::Registry as RegistryTrait,
    types::model::{Accessibility, Entity, EntityType, Model, Reference, Specialization},
};

use crate::registry::Error;

pub struct Registry {
    config: Config,
}

impl Registry {
    pub fn new(config: Config) -> Result<Self, Error> {
        Ok(Self {
            config,
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
            let path = Path::new(&self.config.path);
            if !path.exists() {
                Err(Error::UnableToGetModels {
                    message: format!("Path not found: {}", path.display()),
                })?;
            }

            let entries = fs::read_dir(path).map_err(|error| Error::UnableToGetModels {
                message: error.to_string(),
            })?;

            let mut models = Vec::new();
            for entry in entries.flatten() {
                let path = entry.path();
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                if !file_type.is_dir() {
                    continue;
                }
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                models.push(self.model(name));
            }

            Ok(models)
        })
    }
}

impl Registry {
    fn model(
        &self,
        name: &str,
    ) -> Model {
        let path = Path::new(&self.config.path).join(name);
        let identifier = name.to_string();
        Model {
            identifier: identifier.clone(),
            entities: vec![
                self.entity(EntityType::Registry, &self.config.identifier, &self.config.name),
                self.entity(EntityType::Backend, &self.config.backend_identifier, &self.config.backend_identifier),
                self.entity(EntityType::Vendor, &self.config.identifier, &self.config.name),
                self.entity(EntityType::Family, &self.config.identifier, &self.config.name),
                self.entity(EntityType::Variant, &identifier, &identifier),
            ],
            specializations: vec![Specialization::Chat],
            number_of_parameters: None,
            quantization: None,
            accessibility: Accessibility::Local {
                reference: Reference::Local {
                    path: path.to_string_lossy().to_string(),
                },
            },
        }
    }

    fn entity(
        &self,
        r#type: EntityType,
        identifier: &str,
        name: &str,
    ) -> Entity {
        Entity {
            r#type,
            identifier: identifier.to_string(),
            parent_identifier: None,
            name: name.to_string(),
            description: None,
            version: None,
            icons: vec![],
        }
    }
}
