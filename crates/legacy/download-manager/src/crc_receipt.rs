use std::path::{Path, PathBuf};

use kiban::{fs, time::SystemTime};
use serde::{Deserialize, Serialize};

use crate::Crc32c;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrcReceipt {
    version: u8,
    crc: Crc32c,
    file_size: u64,
    modified_unix_seconds: u64,
    modified_nanos: u32,
}

impl CrcReceipt {
    pub fn path_for(destination: &Path) -> PathBuf {
        PathBuf::from(format!("{}.crc", destination.display()))
    }

    pub async fn for_file(
        path: &Path,
        crc: &Crc32c,
    ) -> Option<Self> {
        let file_size = fs::asyn::file_length(path).await.ok()?;
        let modified = fs::asyn::file_modified(path).await.ok()?.duration_since(SystemTime::UNIX_EPOCH).ok()?;
        Some(Self {
            version: 1,
            crc: crc.clone(),
            file_size,
            modified_unix_seconds: modified.as_secs(),
            modified_nanos: modified.subsec_nanos(),
        })
    }

    pub async fn load(destination: &Path) -> Option<Self> {
        serde_json::from_str(&fs::asyn::read_to_string(Self::path_for(destination)).await.ok()?).ok()
    }
}
