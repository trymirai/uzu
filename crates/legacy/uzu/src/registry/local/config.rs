use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use shoji::types::model::{Model, ModelSpecialization};
use uzu_engine::engine::{ModelType, resolve_model_type};

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
        let resolver: ModelResolver = Arc::new(Self::resolve_model);
        Self::new(
            "lalamo".to_string(),
            backend_identifier,
            backend_version,
            "Lalamo".to_string(),
            models_path.to_string_lossy().to_string(),
            Some(resolver),
        )
    }

    pub fn local(
        backend_identifier: String,
        backend_version: String,
        path: String,
    ) -> Self {
        let resolver: ModelResolver = Arc::new(Self::resolve_model);
        Self::new("local".to_string(), backend_identifier, backend_version, "Local".to_string(), path, Some(resolver))
    }

    fn resolve_model_specialization(model_path: &Path) -> Result<ModelSpecialization, RegistryError> {
        resolve_model_type(model_path)
            .map(|model_type| match model_type {
                ModelType::LanguageModel => ModelSpecialization::Chat {},
                ModelType::Classifier => ModelSpecialization::Classification {},
            })
            .map_err(|error| RegistryError::UnableToGetModels {
                message: format!("Unable to resolve specialization for {}: {error}", model_path.display()),
            })
    }

    fn resolve_model(mut model: Model) -> Result<Model, RegistryError> {
        let model_path = model.filesystem_path().ok_or_else(|| RegistryError::UnableToGetModels {
            message: format!("Filesystem model {} has no path", model.identifier),
        })?;
        let model_path = PathBuf::from(model_path);
        let specialization = Self::resolve_model_specialization(&model_path)?;
        model.specializations = vec![specialization];
        Ok(model)
    }
}
