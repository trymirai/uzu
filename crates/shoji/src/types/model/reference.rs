use serde::{Deserialize, Serialize};

use crate::types::model::{File, Repository};

#[bindings::export(Enum, name = "ModelReference")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Reference {
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

impl Reference {
    pub fn name(&self) -> String {
        match self {
            Reference::Mirai {
                ..
            } => "mirai".to_string(),
            Reference::HuggingFace {
                ..
            } => "huggingface".to_string(),
            Reference::Local {
                ..
            } => "local".to_string(),
        }
    }
}
