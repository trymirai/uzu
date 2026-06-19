use std::{path::PathBuf, sync::Arc};

use backend_uzu::engine::inference::resolve_model_specialization;
use shoji::types::model::Model;

use crate::registry::RegistryError;

pub type ModelResolver = Arc<dyn Fn(Model) -> Result<Model, RegistryError> + Send + Sync>;

#[derive(Clone)]
pub struct Config {
    pub identifier: String,
    pub backend_identifier: String,
    pub backend_version: String,
    pub name: String,
    pub path: String,
    pub resolver: Option<ModelResolver>,
}

impl Config {
    pub fn new(
        identifier: String,
        backend_identifier: String,
        backend_version: String,
        name: String,
        path: String,
        resolver: Option<ModelResolver>,
    ) -> Self {
        Self {
            identifier,
            backend_identifier,
            backend_version,
            name,
            path,
            resolver,
        }
    }

    pub fn lalamo(
        backend_identifier: String,
        backend_version: String,
        path: String,
    ) -> Self {
        let models_path = PathBuf::from(&path).join("models");
        let resolver: ModelResolver = Arc::new(move |mut model: Model| -> Result<Model, RegistryError> {
            let model_path = model.local_external_path().ok_or_else(|| RegistryError::UnableToGetModels {
                message: format!("Local model {} has no local path", model.identifier),
            })?;
            let model_path = PathBuf::from(model_path);
            let specialization =
                resolve_model_specialization(&model_path).map_err(|error| RegistryError::UnableToGetModels {
                    message: format!("Unable to resolve specialization for {}: {error}", model_path.display()),
                })?;
            model.specializations = vec![specialization];
            Ok(model)
        });
        Self::new(
            "lalamo".to_string(),
            backend_identifier,
            backend_version,
            "Lalamo".to_string(),
            models_path.to_string_lossy().to_string(),
            Some(resolver),
        )
    }
}
