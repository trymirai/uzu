use std::sync::Arc;

use shoji::types::basic::{File, HashMethod};

use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct ServedFile {
    pub file: File,
    pub bytes: Arc<[u8]>,
}

impl ServedFile {
    pub fn crc32c(&self) -> Result<String> {
        self.file
            .hashes
            .iter()
            .find(|hash| hash.method == HashMethod::CRC32C)
            .map(|hash| hash.value.clone())
            .ok_or_else(|| Error::MissingCrc32c {
                name: self.file.name.clone(),
            })
    }
}
