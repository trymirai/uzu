use std::{
    io::{Error as IoError, ErrorKind},
    path::{Path, PathBuf},
};

use base64::Engine;
use kiban::{fs, time::SystemTime};
use serde::{Deserialize, Serialize};

const CRC_CACHE_VERSION: u8 = 1;
const CRC_READ_CHUNK_SIZE: u64 = 8 * 1024 * 1024;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
struct CrcCacheReceipt {
    version: u8,
    crc: String,
    file_size: u64,
    modified_unix_seconds: u64,
    modified_nanos: u32,
}

pub async fn calculate_and_verify_crc(
    file_path: &Path,
    expected_crc32c_base64: &str,
) -> Result<bool, IoError> {
    let Some(expected_crc) = decode_crc32c(expected_crc32c_base64) else {
        return Ok(false);
    };

    let file_len = kiban::fs::asyn::file_length(file_path).await?;
    let mut offset = 0;
    let mut actual_crc = 0;
    while offset < file_len {
        let end = offset.saturating_add(CRC_READ_CHUNK_SIZE).min(file_len);
        let chunk = kiban::fs::asyn::read_range(file_path, offset..end).await?;
        if chunk.len() != (end - offset) as usize {
            return Err(IoError::new(ErrorKind::UnexpectedEof, "file changed during CRC verification"));
        }
        actual_crc = crc32c::crc32c_append(actual_crc, &chunk);
        offset = end;
    }

    Ok(actual_crc == expected_crc)
}

pub async fn crc_cache_matches(
    file_path: &Path,
    expected_crc32c_base64: &str,
) -> bool {
    if decode_crc32c(expected_crc32c_base64).is_none() {
        return false;
    }

    let Ok(cache_contents) = kiban::fs::asyn::read_to_string(crc_path_for_file(file_path)).await else {
        return false;
    };
    let Ok(cached_receipt) = serde_json::from_str::<CrcCacheReceipt>(&cache_contents) else {
        return false;
    };
    let Some(current_receipt) = CrcCacheReceipt::from_file(file_path, expected_crc32c_base64).await else {
        return false;
    };

    cached_receipt == current_receipt
}

pub async fn save_crc_file(
    file_path: &Path,
    crc_value: &str,
) -> Result<(), IoError> {
    let Some(receipt) = CrcCacheReceipt::from_file(file_path, crc_value).await else {
        return Ok(());
    };
    let receipt_json = serde_json::to_vec(&receipt).map_err(IoError::other)?;
    fs::asyn::write(crc_path_for_file(file_path), receipt_json).await
}

pub fn crc_path_for_file(file_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.crc", file_path.display()))
}

impl CrcCacheReceipt {
    async fn from_file(
        file_path: &Path,
        crc: &str,
    ) -> Option<Self> {
        if !fs::asyn::is_file(file_path).await {
            return None;
        }

        let file_size = fs::asyn::file_length(file_path).await.ok()?;
        let modified = fs::asyn::file_modified(file_path).await.ok()?.duration_since(SystemTime::UNIX_EPOCH).ok()?;
        Some(Self {
            version: CRC_CACHE_VERSION,
            crc: crc.to_string(),
            file_size,
            modified_unix_seconds: modified.as_secs(),
            modified_nanos: modified.subsec_nanos(),
        })
    }
}

fn decode_crc32c(expected_crc32c_base64: &str) -> Option<u32> {
    let expected_bytes = base64::engine::general_purpose::STANDARD.decode(expected_crc32c_base64).ok()?;

    if expected_bytes.len() != 4 {
        return None;
    }

    Some(u32::from_be_bytes([expected_bytes[0], expected_bytes[1], expected_bytes[2], expected_bytes[3]]))
}
