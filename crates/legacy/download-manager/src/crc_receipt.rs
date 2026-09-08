use std::{
    io,
    path::{Path, PathBuf},
};

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
    pub async fn matches(
        destination: &Path,
        crc: &Crc32c,
    ) -> bool {
        matches!(
            (Self::load(destination).await, Self::for_file(destination, crc).await),
            (Some(saved), Some(current)) if saved == current
        )
    }

    pub async fn save(
        destination: &Path,
        crc: &Crc32c,
    ) -> Result<(), io::Error> {
        let Some(receipt) = Self::for_file(destination, crc).await else {
            return Ok(());
        };
        fs::asyn::write(Self::path_for(destination), serde_json::to_vec(&receipt).map_err(io::Error::other)?).await
    }

    pub async fn exists(destination: &Path) -> bool {
        fs::asyn::is_file(Self::path_for(destination)).await
    }

    pub async fn remove(destination: &Path) {
        let _ = fs::asyn::remove_file(Self::path_for(destination)).await;
    }

    async fn load(destination: &Path) -> Option<Self> {
        serde_json::from_str(&fs::asyn::read_to_string(Self::path_for(destination)).await.ok()?).ok()
    }

    async fn for_file(
        destination: &Path,
        crc: &Crc32c,
    ) -> Option<Self> {
        let file_size = fs::asyn::file_length(destination).await.ok()?;
        let modified = fs::asyn::file_modified(destination).await.ok()?.duration_since(SystemTime::UNIX_EPOCH).ok()?;
        Some(Self {
            version: 1,
            crc: crc.clone(),
            file_size,
            modified_unix_seconds: modified.as_secs(),
            modified_nanos: modified.subsec_nanos(),
        })
    }

    fn path_for(destination: &Path) -> PathBuf {
        PathBuf::from(format!("{}.crc", destination.display()))
    }
}
