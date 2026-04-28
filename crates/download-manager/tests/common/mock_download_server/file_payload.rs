use std::sync::Arc;

use reqwest::Url;
use shoji::types::basic::File;

#[derive(Clone, Debug)]
pub struct FilePayload {
    pub file: File,
    pub bytes: Arc<[u8]>,
    pub last_modified: String,
}

impl FilePayload {
    pub fn path(&self) -> String {
        let url = Url::parse(&self.file.url).unwrap();
        url.path().to_string()
    }

    pub fn crc32c(&self) -> String {
        self.file.crc32c().unwrap()
    }

    pub fn corrupt_bytes(&self) -> Arc<[u8]> {
        let mut bytes = self.bytes.to_vec();
        if let Some(first_byte) = bytes.first_mut() {
            *first_byte = first_byte.wrapping_add(1);
        }
        Arc::from(bytes)
    }
}
