use std::{
    io::{Error as IoError, ErrorKind},
    path::Path,
};

use base64::Engine;
use kiban::fs;
use serde::{Deserialize, Serialize};

use crate::crc_receipt::CrcReceipt;

const READ_CHUNK_SIZE: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Crc32c(String);

impl From<String> for Crc32c {
    fn from(encoded: String) -> Self {
        Self(encoded)
    }
}

impl Crc32c {
    pub async fn verify(
        &self,
        path: &Path,
    ) -> Result<bool, IoError> {
        let Some(expected) = self.decode() else {
            return Ok(false);
        };
        let length = fs::asyn::file_length(path).await?;
        let mut checksum = 0_u32;
        let mut offset = 0_u64;
        while offset < length {
            let end = offset.saturating_add(READ_CHUNK_SIZE).min(length);
            let chunk = fs::asyn::read_range(path, offset..end).await?;
            if chunk.len() as u64 != end - offset {
                return Err(IoError::new(ErrorKind::UnexpectedEof, "file changed during CRC verification"));
            }
            checksum = ::crc32c::crc32c_append(checksum, &chunk);
            offset = end;
        }
        Ok(checksum == expected)
    }

    pub async fn cached_matches(
        &self,
        path: &Path,
    ) -> bool {
        matches!(
            (CrcReceipt::load(path).await, CrcReceipt::for_file(path, self).await),
            (Some(saved), Some(current)) if saved == current
        )
    }

    pub async fn save_receipt(
        &self,
        path: &Path,
    ) -> Result<(), IoError> {
        let Some(receipt) = CrcReceipt::for_file(path, self).await else {
            return Ok(());
        };
        fs::asyn::write(CrcReceipt::path_for(path), serde_json::to_vec(&receipt).map_err(IoError::other)?).await
    }

    fn decode(&self) -> Option<u32> {
        let bytes = base64::engine::general_purpose::STANDARD.decode(&self.0).ok()?;
        Some(u32::from_be_bytes(bytes.try_into().ok()?))
    }
}
