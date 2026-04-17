use serde::{Deserialize, Serialize};

use crate::registry::types::{File, Repository};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Reference {
    HuggingFace {
        repository: Repository,
    },
    Mirai {
        toolchain_version: String,
        repository: Option<Repository>,
        source_repository: Option<Repository>,
        files: Vec<File>,
    },
}

impl Reference {
    pub fn r#type(&self) -> String {
        match self {
            Reference::HuggingFace {
                ..
            } => "huggingface".to_string(),
            Reference::Mirai {
                ..
            } => "mirai".to_string(),
        }
    }
}
