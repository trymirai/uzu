use serde::{Deserialize, Serialize};

use crate::registry::types::{Accessibility, Entity, Reference, Specialization};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Model {
    pub entities: Vec<Entity>,
    pub specializations: Vec<Specialization>,
    pub number_of_parameters: Option<u64>,
    pub accessibility: Accessibility,
}

impl Model {
    pub fn identifier(&self) -> String {
        self.entities.iter().map(|entity| entity.identifier.clone()).collect::<Vec<String>>().join(":")
    }

    pub fn cache_identifier(&self) -> String {
        self.identifier().replace(":", "-")
    }

    pub fn is_local(&self) -> bool {
        matches!(self.accessibility, Accessibility::Local { .. })
    }

    pub fn is_remote(&self) -> bool {
        matches!(self.accessibility, Accessibility::Remote { .. })
    }

    pub fn provider_identifier(&self) -> String {
        match &self.accessibility {
            Accessibility::Local {
                provider_identifier,
                ..
            } => provider_identifier.clone(),
            Accessibility::Remote {
                provider_identifier,
                ..
            } => provider_identifier.clone(),
        }
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
            },
            Accessibility::Remote {
                source_repository,
                ..
            } => {
                let mut result = vec![];
                if let Some(source_repository) = source_repository {
                    result.push(source_repository.identifier.clone());
                }
                result
            },
        }
    }

    pub fn reference_type(&self) -> Option<String> {
        match &self.accessibility {
            Accessibility::Local {
                reference,
                ..
            } => Some(reference.r#type()),
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
            },
            Accessibility::Remote {
                ..
            } => None,
        }
    }
}
