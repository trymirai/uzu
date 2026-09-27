use serde::{Deserialize, Serialize};

use crate::types::basic::{File, Repository};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelSource {
    #[serde(rename = "mirai")]
    Registry {
        toolchain_version: String,
        repository: Option<Repository>,
        source_repository: Option<Repository>,
        files: Vec<File>,
    },
    #[serde(rename = "local")]
    Filesystem {
        path: String,
    },
}
