use serde::{Deserialize, Serialize};

use crate::types::model::{
    ModelAccessibility, ModelEntity, ModelEntityType, ModelQuantization, ModelReference, ModelSpecialization,
};

#[bindings::export(ClassCloneable)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Model {
    pub identifier: String,
    pub entities: Vec<ModelEntity>,
    pub specializations: Vec<ModelSpecialization>,
    pub number_of_parameters: Option<i64>,
    pub quantization: Option<ModelQuantization>,
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
    #[bindings::export(Getter)]
    pub fn registry_entity(&self) -> Option<ModelEntity> {
        self.entity(ModelEntityType::Registry)
    }

    #[bindings::export(Getter)]
    pub fn backend_entity(&self) -> Option<ModelEntity> {
        self.entity(ModelEntityType::Backend)
    }

    #[bindings::export(Getter)]
    pub fn vendor_entity(&self) -> Option<ModelEntity> {
        self.entity(ModelEntityType::Vendor)
    }

    #[bindings::export(Getter)]
    pub fn family_entity(&self) -> Option<ModelEntity> {
        self.entity(ModelEntityType::Family)
    }

    #[bindings::export(Getter)]
    pub fn variant_entity(&self) -> Option<ModelEntity> {
        self.entity(ModelEntityType::Variant)
    }
}

impl Model {
    fn entity(
        &self,
        r#type: ModelEntityType,
    ) -> Option<ModelEntity> {
        self.entities.iter().find(|entity| entity.r#type == r#type).map(|entity| entity.clone())
    }
}
