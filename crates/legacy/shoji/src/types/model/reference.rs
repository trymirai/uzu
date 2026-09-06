use serde::{Deserialize, Serialize};

use crate::types::basic::{File, Repository};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelReference {
    Mirai {
        toolchain_version: String,
        repository: Option<Repository>,
        source_repository: Option<Repository>,
        files: Vec<File>,
    },
    HuggingFace {
        repository: Repository,
    },
    Local {
        path: String,
    },
}

impl ModelReference {
    pub fn name(&self) -> String {
        match self {
            ModelReference::Mirai {
                ..
            } => "mirai".to_string(),
            ModelReference::HuggingFace {
                ..
            } => "huggingface".to_string(),
            ModelReference::Local {
                ..
            } => "local".to_string(),
        }
    }
}
