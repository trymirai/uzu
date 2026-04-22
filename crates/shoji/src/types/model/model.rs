use serde::{Deserialize, Serialize};

use crate::types::model::{
    Entity, EntityType, ModelAccessibility, ModelQuantization, ModelReference, ModelSpecialization,
};

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Model {
    pub identifier: String,
    pub entities: Vec<Entity>,
    pub specializations: Vec<ModelSpecialization>,
    pub number_of_parameters: Option<i64>,
    pub quantization: Option<ModelQuantization>,
    pub accessibility: ModelAccessibility,
}

impl Model {
    pub fn identifier(&self) -> String {
        self.identifier.clone()
    }

    pub fn cache_identifier(&self) -> String {
        self.identifier.replace(":", "-").replace("/", "-")
    }

    pub fn is_local(&self) -> bool {
        matches!(self.accessibility, ModelAccessibility::Local { .. })
    }

    pub fn is_remote(&self) -> bool {
        matches!(self.accessibility, ModelAccessibility::Remote { .. })
    }

    pub fn is_downloadable(&self) -> bool {
        matches!(
            self.accessibility,
            ModelAccessibility::Local {
                reference: ModelReference::Mirai { .. } | ModelReference::HuggingFace { .. }
            }
        )
    }

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

impl Model {
    pub fn registry_entity(&self) -> Option<Entity> {
        self.entity(EntityType::Registry)
    }

    pub fn backend_entity(&self) -> Option<Entity> {
        self.entity(EntityType::Backend)
    }

    pub fn vendor_entity(&self) -> Option<Entity> {
        self.entity(EntityType::Vendor)
    }

    pub fn family_entity(&self) -> Option<Entity> {
        self.entity(EntityType::Family)
    }

    pub fn variant_entity(&self) -> Option<Entity> {
        self.entity(EntityType::Variant)
    }

    fn entity(
        &self,
        r#type: EntityType,
    ) -> Option<Entity> {
        self.entities.iter().find(|entity| entity.r#type == r#type).map(|entity| entity.clone())
    }
}
