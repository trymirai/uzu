use serde::{Deserialize, Serialize};

use crate::types::model::{Hash, HashMethod};

#[bindings::export(Struct, name = "File")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct File {
    pub url: String,
    pub name: String,
    pub size: i64,
    pub hashes: Vec<Hash>,
}

impl File {
    pub fn crc32c(&self) -> Option<String> {
        self.hashes.iter().find(|hash| hash.method == HashMethod::CRC32C).map(|hash| hash.value.clone())
    }
}
