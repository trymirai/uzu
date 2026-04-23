mod config;

use std::{fs, future::Future, path::Path, pin::Pin};

pub use config::Config;
use shoji::{
    traits::Registry as RegistryTrait,
    types::model::{Model, ModelAccessibility, ModelReference, ModelSpecialization},
};

use crate::registry::RegistryError;

pub struct Registry {
    config: Config,
}

impl Registry {
    pub fn new(config: Config) -> Result<Self, RegistryError> {
        Ok(Self {
            config,
        })
    }
}

impl RegistryTrait for Registry {
    type Error = RegistryError;

    fn indentifier(&self) -> String {
        self.config.identifier.clone()
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, RegistryError>> + Send + '_>> {
        Box::pin(async {
            let path = Path::new(&self.config.path);
            if !path.exists() {
                Err(RegistryError::UnableToGetModels {
                    message: format!("Path not found: {}", path.display()),
                })?;
            }

            let entries = fs::read_dir(path).map_err(|error| RegistryError::UnableToGetModels {
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

                let model = self.model(name);
                let model = match self.config.resolver.as_ref() {
                    Some(resolver) => resolver(model),
                    None => Some(model),
                };
                if let Some(model) = model {
                    models.push(model);
                }
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
        Model::external(
            identifier.clone(),
            self.config.identifier.clone(),
            self.config.name.clone(),
            self.config.backend_identifier.clone(),
            self.config.backend_identifier.clone(),
            self.config.backend_version.clone(),
            vec![ModelSpecialization::Chat],
            ModelAccessibility::Local {
                reference: ModelReference::Local {
                    path: path.to_string_lossy().to_string(),
                },
            },
        )
    }
}
