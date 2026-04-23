use serde::{Deserialize, Serialize};

use crate::types::{
    basic::Metadata,
    model::{
        ModelAccessibility, ModelBackend, ModelFamily, ModelProperties, ModelQuantization, ModelReference,
        ModelRegistry, ModelSpecialization,
    },
};

#[bindings::export(ClassCloneable)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Model {
    #[serde(rename = "id")]
    pub identifier: String,
    pub registry: ModelRegistry,
    pub backends: Vec<ModelBackend>,
    pub family: Option<ModelFamily>,
    pub properties: Option<ModelProperties>,
    pub quantization: Option<ModelQuantization>,
    pub specializations: Vec<ModelSpecialization>,
    pub accessibility: ModelAccessibility,
}

#[bindings::export(Implementation)]
impl Model {
    #[bindings::export(Getter)]
    pub fn cache_identifier(&self) -> String {
        self.identifier.replace(":", "-").replace("/", "-")
    }

    #[bindings::export(Getter)]
    pub fn is_local(&self) -> bool {
        matches!(self.accessibility, ModelAccessibility::Local { .. })
    }

    #[bindings::export(Getter)]
    pub fn is_remote(&self) -> bool {
        matches!(self.accessibility, ModelAccessibility::Remote { .. })
    }

    #[bindings::export(Getter)]
    pub fn is_downloadable(&self) -> bool {
        matches!(
            self.accessibility,
            ModelAccessibility::Local {
                reference: ModelReference::Mirai { .. } | ModelReference::HuggingFace { .. }
            }
        )
    }

    #[bindings::export(Getter)]
    pub fn repo_ids(&self) -> Vec<String> {
        match &self.accessibility {
            ModelAccessibility::Local {
                reference,
                ..
            } => match reference {
                ModelReference::Mirai {
                    repository,
                    source_repository,
                    ..
                } => {
                    let mut result = vec![];
                    if let Some(repository) = repository {
                        result.push(repository.identifier.clone());
                    }
                    if let Some(source_repository) = source_repository {
                        result.push(source_repository.identifier.clone());
                    }
                    result
                },
                ModelReference::HuggingFace {
                    repository,
                } => vec![repository.identifier.clone()],
                ModelReference::Local {
                    ..
                } => vec![],
            },
            ModelAccessibility::Remote {
                repository,
                ..
            } => {
                let mut result = vec![];
                if let Some(repository) = repository {
                    result.push(repository.identifier.clone());
                }
                result
            },
        }
    }

    #[bindings::export(Getter)]
    pub fn local_external_path(&self) -> Option<String> {
        match &self.accessibility {
            ModelAccessibility::Local {
                reference,
                ..
            } => match reference {
                ModelReference::Mirai {
                    ..
                } => None,
                ModelReference::HuggingFace {
                    ..
                } => None,
                ModelReference::Local {
                    path,
                } => Some(path.clone()),
            },
            ModelAccessibility::Remote {
                ..
            } => None,
        }
    }

    #[bindings::export(Getter)]
    pub fn reference_name(&self) -> Option<String> {
        match &self.accessibility {
            ModelAccessibility::Local {
                reference,
                ..
            } => Some(reference.name()),
            ModelAccessibility::Remote {
                ..
            } => None,
        }
    }

    #[bindings::export(Getter)]
    pub fn checkpoint_version(&self) -> Option<String> {
        match &self.accessibility {
            ModelAccessibility::Local {
                reference,
                ..
            } => match reference {
                ModelReference::Mirai {
                    toolchain_version,
                    ..
                } => Some(toolchain_version.clone()),
                ModelReference::HuggingFace {
                    repository,
                } => Some(repository.commit_hash.clone()),
                ModelReference::Local {
                    ..
                } => None,
            },
            ModelAccessibility::Remote {
                ..
            } => None,
        }
    }
}

#[bindings::export(Implementation)]
impl Model {
    #[bindings::export(Factory)]
    pub fn external(
        identifier: String,
        registry_identifier: String,
        registry_name: String,
        backend_identifier: String,
        backend_name: String,
        backend_version: String,
        specializations: Vec<ModelSpecialization>,
        accessibility: ModelAccessibility,
    ) -> Self {
        let registry = ModelRegistry {
            identifier: registry_identifier,
            metadata: Metadata::external(registry_name),
        };
        let backend = ModelBackend {
            identifier: backend_identifier.clone(),
            version: backend_version,
            metadata: Metadata::external(backend_name),
        };
        Self {
            identifier,
            registry,
            backends: vec![backend],
            family: None,
            properties: None,
            quantization: None,
            specializations,
            accessibility,
        }
    }
}
