use std::sync::Arc;

use shoji::types::basic::{File, HashMethod};

#[derive(Clone, Debug)]
pub struct ServedFile {
    pub file: File,
    pub bytes: Arc<[u8]>,
}

impl ServedFile {
    pub fn crc32c(&self) -> String {
        self.file
            .hashes
            .iter()
            .find(|hash| hash.method == HashMethod::CRC32C)
            .map(|hash| hash.value.clone())
            .expect("mock registry files must always have CRC32C hashes")
    }
}
