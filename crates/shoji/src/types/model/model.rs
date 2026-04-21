use serde::{Deserialize, Serialize};

use crate::types::model::{Accessibility, Entity, EntityType, Quantization, Reference, Specialization};

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Model {
    pub identifier: String,
    pub entities: Vec<Entity>,
    pub specializations: Vec<Specialization>,
    pub number_of_parameters: Option<i64>,
    pub quantization: Option<Quantization>,
    pub accessibility: Accessibility,
}

impl Model {
    pub fn identifier(&self) -> String {
        self.identifier.clone()
    }

    pub fn cache_identifier(&self) -> String {
        self.identifier.replace(":", "-").replace("/", "-")
    }

    pub fn is_local(&self) -> bool {
        matches!(self.accessibility, Accessibility::Local { .. })
    }

    pub fn is_remote(&self) -> bool {
        matches!(self.accessibility, Accessibility::Remote { .. })
    }

    pub fn is_downloadable(&self) -> bool {
        matches!(
            self.accessibility,
            Accessibility::Local {
                reference: Reference::Mirai { .. } | Reference::HuggingFace { .. }
            }
        )
    }

    pub fn repo_ids(&self) -> Vec<String> {
        match &self.accessibility {
            Accessibility::Local {
                reference,
                ..
            } => match reference {
                Reference::Mirai {
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
                Reference::HuggingFace {
                    repository,
                } => vec![repository.identifier.clone()],
                Reference::Local {
                    ..
                } => vec![],
            },
            Accessibility::Remote {
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
            Accessibility::Local {
                reference,
                ..
            } => match reference {
                Reference::Mirai {
                    ..
                } => None,
                Reference::HuggingFace {
                    ..
                } => None,
                Reference::Local {
                    path,
                } => Some(path.clone()),
            },
            Accessibility::Remote {
                ..
            } => None,
        }
    }

    pub fn reference_name(&self) -> Option<String> {
        match &self.accessibility {
            Accessibility::Local {
                reference,
                ..
            } => Some(reference.name()),
            Accessibility::Remote {
                ..
            } => None,
        }
    }

    pub fn checkpoint_version(&self) -> Option<String> {
        match &self.accessibility {
            Accessibility::Local {
                reference,
                ..
            } => match reference {
                Reference::Mirai {
                    toolchain_version,
                    ..
                } => Some(toolchain_version.clone()),
                Reference::HuggingFace {
                    repository,
                } => Some(repository.commit_hash.clone()),
                Reference::Local {
                    ..
                } => None,
            },
            Accessibility::Remote {
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
