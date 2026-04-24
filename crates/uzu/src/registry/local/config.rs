use std::{path::PathBuf, sync::Arc};

use backend_uzu::inference::resolve_model_metadata;
use shoji::types::model::Model;

pub type ModelResolver = Arc<dyn Fn(Model) -> Option<Model> + Send + Sync>;

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
        let resolver: ModelResolver = {
            let backend_version = backend_version.clone();
            Arc::new(move |mut model: Model| -> Option<Model> {
                let model_path = PathBuf::from(model.local_external_path()?);
                let metadata = resolve_model_metadata(&model_path)?;
                if !is_lalamo_version_supported(&metadata.toolchain_version, &backend_version) {
                    return None;
                }
                model.specializations = vec![metadata.specialization];
                Some(model)
            })
        };
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

fn is_lalamo_version_supported(
    model_version: &str,
    backend_version: &str,
) -> bool {
    let parse_version = |value: &str| -> Option<(u32, u32, u32)> {
        let mut parts = value.split('.').map(|part| part.parse::<u32>().ok());
        let major = parts.next().flatten()?;
        let minor = parts.next().flatten()?;
        let patch = parts.next().flatten()?;
        Some((major, minor, patch))
    };
    let Some((model_major, model_minor, model_patch)) = parse_version(model_version) else {
        return false;
    };
    let Some((backend_major, backend_minor, backend_patch)) = parse_version(backend_version) else {
        return false;
    };
    model_major == backend_major && model_minor == backend_minor && model_patch <= backend_patch
}
