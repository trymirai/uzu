use std::{
    fs::{self, File},
    io::{Error as IoError, Read},
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

use base64::Engine;
use serde::{Deserialize, Serialize};

const CHUNK_SIZE: usize = 8 * 1024 * 1024;
const CRC_CACHE_VERSION: u8 = 1;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
struct CrcCacheReceipt {
    version: u8,
    crc: String,
    file_size: u64,
    modified_unix_seconds: u64,
    modified_nanos: u32,
}

pub fn calculate_and_verify_crc(
    file_path: &Path,
    expected_crc32c_base64: &str,
) -> Result<bool, IoError> {
    let Some(expected_crc) = decode_crc32c(expected_crc32c_base64) else {
        return Ok(false);
    };

    let mut file = File::open(file_path)?;
    let mut buffer = vec![0_u8; CHUNK_SIZE];
    let mut crc = 0_u32;

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        crc = crc32c::crc32c_append(crc, &buffer[..bytes_read]);
    }

    Ok(crc == expected_crc)
}

pub fn crc_cache_matches(
    file_path: &Path,
    expected_crc32c_base64: &str,
) -> bool {
    if decode_crc32c(expected_crc32c_base64).is_none() {
        return false;
    }

    let Ok(cache_contents) = fs::read_to_string(crc_path_for_file(file_path)) else {
        return false;
    };
    let Ok(cached_receipt) = serde_json::from_str::<CrcCacheReceipt>(&cache_contents) else {
        return false;
    };
    let Some(current_receipt) = CrcCacheReceipt::from_file(file_path, expected_crc32c_base64) else {
        return false;
    };

    cached_receipt == current_receipt
}

pub fn save_crc_file(
    file_path: &Path,
    crc_value: &str,
) -> Result<(), IoError> {
    let Some(receipt) = CrcCacheReceipt::from_file(file_path, crc_value) else {
        return Ok(());
    };
    let receipt_json = serde_json::to_vec(&receipt).map_err(IoError::other)?;
    std::fs::write(crc_path_for_file(file_path), receipt_json)
}

pub fn crc_path_for_file(file_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.crc", file_path.display()))
}

impl CrcCacheReceipt {
    fn from_file(
        file_path: &Path,
        crc: &str,
    ) -> Option<Self> {
        let metadata = fs::metadata(file_path).ok()?;
        if !metadata.is_file() {
            return None;
        }
        let modified = metadata.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;

        Some(Self {
            version: CRC_CACHE_VERSION,
            crc: crc.to_string(),
            file_size: metadata.len(),
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
