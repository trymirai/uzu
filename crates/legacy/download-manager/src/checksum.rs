use std::{
    io::{Error as IoError, ErrorKind},
    path::Path,
};

use base64::Engine;
use kiban::fs;
use serde::{Deserialize, Serialize};
use sha1::Sha1;
use sha2::{Digest, Sha256};

const READ_CHUNK_SIZE: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Checksum {
    Crc32c(String),
    Sha256(String),
    GitBlobSha1(String),
}

impl Checksum {
    pub fn algorithm(&self) -> &'static str {
        match self {
            Self::Crc32c(_) => "CRC32C",
            Self::Sha256(_) => "SHA-256",
            Self::GitBlobSha1(_) => "Git blob SHA-1",
        }
    }

    pub async fn verify(
        &self,
        path: &Path,
    ) -> Result<bool, IoError> {
        let length = fs::asyn::file_length(path).await?;
        match self {
            Self::Crc32c(expected) => {
                let Some(expected) = base64::engine::general_purpose::STANDARD
                    .decode(expected)
                    .ok()
                    .and_then(|bytes| Some(u32::from_be_bytes(bytes.try_into().ok()?)))
                else {
                    return Ok(false);
                };
                let actual = fold_chunks(path, length, 0_u32, |checksum, chunk| {
                    *checksum = ::crc32c::crc32c_append(*checksum, chunk)
                })
                .await?;
                Ok(actual == expected)
            },
            Self::Sha256(expected) => {
                let digest =
                    fold_chunks(path, length, Sha256::new(), |hasher, chunk| hasher.update(chunk)).await?.finalize();
                Ok(hex(&digest) == expected.to_ascii_lowercase())
            },
            Self::GitBlobSha1(expected) => {
                let mut hasher = Sha1::new();
                hasher.update(format!("blob {length}\0"));
                let digest = fold_chunks(path, length, hasher, |hasher, chunk| hasher.update(chunk)).await?.finalize();
                Ok(hex(&digest) == expected.to_ascii_lowercase())
            },
        }
    }
}

async fn fold_chunks<S>(
    path: &Path,
    length: u64,
    mut state: S,
    mut update: impl FnMut(&mut S, &[u8]),
) -> Result<S, IoError> {
    let mut offset = 0_u64;
    while offset < length {
        let end = offset.saturating_add(READ_CHUNK_SIZE).min(length);
        let chunk = fs::asyn::read_range(path, offset..end).await?;
        if chunk.len() as u64 != end - offset {
            return Err(IoError::new(ErrorKind::UnexpectedEof, "file changed during checksum verification"));
        }
        update(&mut state, &chunk);
        offset = end;
    }
    Ok(state)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
