use serde::{Deserialize, Serialize};

use crate::types::basic::{Hash, HashMethod};

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct File {
    pub url: String,
    pub name: String,
    pub size: i64,
    pub hashes: Vec<Hash>,
}

#[bindings::export(Implementation)]
impl File {
    #[bindings::export(Getter)]
    pub fn crc32c(&self) -> Option<String> {
        self.hashes.iter().find(|hash| hash.method == HashMethod::CRC32C).map(|hash| hash.value.clone())
    }

    #[bindings::export(Getter)]
    pub fn md5(&self) -> Option<String> {
        self.hashes.iter().find(|hash| hash.method == HashMethod::MD5).map(|hash| hash.value.clone())
    }
}
