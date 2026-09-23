mod config;

use std::{fs, future::Future, path::Path, pin::Pin};

pub use config::Config;
use hanashi::chat::EncodingConfig;
use shoji::{
    traits::Registry as RegistryTrait,
    types::{
        basic::Value,
        model::{Model, ModelAccessibility, ModelReference, ModelSpecialization},
    },
};
use uzu_engine::engine::{ModelType, resolve_model_type};

use crate::registry::RegistryError;

pub struct Registry {
    config: Config,
}

impl Registry {
    pub(crate) fn model_at_path(
        path: &Path,
        backend_identifier: String,
        backend_version: String,
    ) -> Result<Model, RegistryError> {
        let registry =
            Self::new(Config::new(backend_identifier, backend_version, path.to_string_lossy().into_owned()))?;
        registry.model(Path::new(&registry.config.path))
    }

    pub fn new(mut config: Config) -> Result<Self, RegistryError> {
        config.path = Path::new(&config.path)
            .canonicalize()
            .map_err(|error| RegistryError::UnableToGetModels {
                message: format!("Unable to open local model path {}: {error}", config.path),
            })?
            .to_string_lossy()
            .into_owned();
        Ok(Self {
            config,
        })
    }
}

impl RegistryTrait for Registry {
    type Error = RegistryError;

    fn identifier(&self) -> String {
        "local".to_string()
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, RegistryError>> + Send + '_>> {
        Box::pin(async {
            let path = Path::new(&self.config.path);
            if path.join("config.json").is_file() {
                return Ok(vec![self.model(path)?]);
            }

            let entries = fs::read_dir(path).map_err(|error| RegistryError::UnableToGetModels {
                message: error.to_string(),
            })?;

            let mut models = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|error| RegistryError::UnableToGetModels {
                    message: error.to_string(),
                })?;
                let path = entry.path();
                let file_type = entry.file_type().map_err(|error| RegistryError::UnableToGetModels {
                    message: error.to_string(),
                })?;
                if !file_type.is_dir() {
                    continue;
                }
                match self.model(&path) {
                    Ok(model) => models.push(model),
                    Err(error) => {
                        tracing::warn!(?error, path = %path.display(), "skipping invalid local model");
                    },
                }
            }

            Ok(models)
        })
    }
}

impl Registry {
    fn model(
        &self,
        path: &Path,
    ) -> Result<Model, RegistryError> {
        let name = path.file_name().and_then(|name| name.to_str()).ok_or_else(|| RegistryError::UnableToGetModels {
            message: format!("Invalid local model path: {}", path.display()),
        })?;
        let specialization = resolve_model_type(path)
            .map(|model_type| match model_type {
                ModelType::LanguageModel => ModelSpecialization::Chat {},
                ModelType::Classifier => ModelSpecialization::Classification {},
            })
            .map_err(|error| RegistryError::UnableToGetModels {
                message: format!("Unable to resolve specialization for {}: {error}", path.display()),
            })?;
        Ok(Model::external(
            name.to_string(),
            self.identifier(),
            "Local".to_string(),
            self.config.backend_identifier.clone(),
            self.config.backend_identifier.clone(),
            self.config.backend_version.clone(),
            vec![specialization],
            ModelAccessibility::Local {
                reference: ModelReference::Local {
                    path: path.to_string_lossy().to_string(),
                },
            },
            load_encoding(path),
        ))
    }
}

fn load_encoding(model_path: &Path) -> Option<Value> {
    let Ok(text) = fs::read_to_string(model_path.join("encoding.json")) else {
        return None;
    };
    let parsed = serde_json::from_str::<serde_json::Value>(&text);
    let encodings: Vec<Value> = match parsed {
        Ok(serde_json::Value::Array(entries)) => entries.into_iter().map(Value::from).collect(),
        Ok(entry) => vec![Value::from(entry)],
        Err(error) => {
            tracing::warn!(?error, path = %model_path.display(), "ignoring invalid encoding.json");
            vec![]
        },
    };
    EncodingConfig::select(&encodings)
}
