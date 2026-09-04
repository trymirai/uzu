use std::{
    io::{Error as IoError, ErrorKind},
    path::{Path, PathBuf},
};

use base64::{Engine, engine::general_purpose::STANDARD};
use crc32c::crc32c_append;
use hex::decode_to_slice;
use kiban::{fs, time::SystemTime};
use serde::{Deserialize, Serialize};
use serde_json::{from_str, to_vec};
use sha1::Sha1;
use sha2::{Digest, Sha256};

use crate::FileCheck;

const INTEGRITY_RECEIPT_VERSION: u8 = 1;
const READ_CHUNK_SIZE: u64 = 8 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
    #[error("invalid expected {algorithm} digest")]
    InvalidExpectedDigest {
        algorithm: &'static str,
    },
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Serialize, Deserialize, PartialEq, Eq)]
struct IntegrityReceipt {
    version: u8,
    file_check: FileCheck,
    #[serde(flatten)]
    fingerprint: FileFingerprint,
}

#[derive(Serialize, Deserialize, PartialEq, Eq)]
struct FileFingerprint {
    file_size: u64,
    modified_unix_seconds: u64,
    modified_nanos: u32,
}

enum StreamingVerifier {
    Crc32c {
        expected: u32,
        actual: u32,
    },
    Sha256 {
        expected: [u8; 32],
        actual: Sha256,
    },
    GitBlobSha1 {
        expected: [u8; 20],
        actual: Sha1,
    },
}

pub async fn verify_file_integrity(
    file_path: &Path,
    file_check: &FileCheck,
) -> Result<bool, VerificationError> {
    if matches!(file_check, FileCheck::None) {
        return Ok(true);
    }

    let file_len = fs::asyn::file_length(file_path).await?;
    let mut verifier = StreamingVerifier::new(file_check, file_len)?;
    let mut offset = 0;
    while offset < file_len {
        let end = offset.saturating_add(READ_CHUNK_SIZE).min(file_len);
        let chunk = fs::asyn::read_range(file_path, offset..end).await?;
        let expected_len = usize::try_from(end - offset)
            .map_err(|_| IoError::new(ErrorKind::InvalidData, "integrity chunk size overflow"))?;
        if chunk.len() != expected_len {
            return Err(IoError::new(ErrorKind::UnexpectedEof, "file changed during integrity verification").into());
        }
        verifier.update(&chunk);
        offset = end;
    }
    if fs::asyn::file_length(file_path).await? != file_len {
        return Err(IoError::new(ErrorKind::UnexpectedEof, "file changed during integrity verification").into());
    }

    Ok(verifier.matches_expected())
}

pub async fn integrity_cache_matches(
    file_path: &Path,
    file_check: &FileCheck,
) -> bool {
    if !expected_digest_is_well_formed(file_check) {
        return false;
    }
    let Ok(cache_contents) = fs::asyn::read_to_string(integrity_receipt_path(file_path)).await else {
        return false;
    };
    let Ok(fingerprint) = FileFingerprint::from_file(file_path).await else {
        return false;
    };

    from_str::<IntegrityReceipt>(&cache_contents).is_ok_and(|cached| {
        cached
            == IntegrityReceipt {
                version: INTEGRITY_RECEIPT_VERSION,
                file_check: file_check.clone(),
                fingerprint,
            }
    })
}

pub async fn save_integrity_cache(
    file_path: &Path,
    file_check: &FileCheck,
) -> Result<(), IoError> {
    if matches!(file_check, FileCheck::None) {
        return Ok(());
    }
    let fingerprint = FileFingerprint::from_file(file_path).await?;
    let receipt = to_vec(&IntegrityReceipt {
        version: INTEGRITY_RECEIPT_VERSION,
        file_check: file_check.clone(),
        fingerprint,
    })
    .map_err(IoError::other)?;
    fs::asyn::write(integrity_receipt_path(file_path), receipt).await
}

pub fn integrity_receipt_path(file_path: &Path) -> PathBuf {
    file_path.with_added_extension("integrity")
}

impl StreamingVerifier {
    fn new(
        file_check: &FileCheck,
        file_len: u64,
    ) -> Result<Self, VerificationError> {
        match file_check {
            FileCheck::CRC(expected) => Ok(Self::Crc32c {
                expected: decode_crc32c(expected).ok_or(VerificationError::InvalidExpectedDigest {
                    algorithm: "crc32c",
                })?,
                actual: 0,
            }),
            FileCheck::Sha256(expected) => Ok(Self::Sha256 {
                expected: decode_hex(expected, "sha256")?,
                actual: Sha256::new(),
            }),
            FileCheck::GitBlobSha1(expected) => {
                let mut actual = Sha1::new();
                actual.update(format!("blob {file_len}\0").as_bytes());
                Ok(Self::GitBlobSha1 {
                    expected: decode_hex(expected, "git_blob_sha1")?,
                    actual,
                })
            },
            FileCheck::None => unreachable!("handled before constructing verifier"),
        }
    }

    fn update(
        &mut self,
        chunk: &[u8],
    ) {
        match self {
            Self::Crc32c {
                actual,
                ..
            } => *actual = crc32c_append(*actual, chunk),
            Self::Sha256 {
                actual,
                ..
            } => actual.update(chunk),
            Self::GitBlobSha1 {
                actual,
                ..
            } => actual.update(chunk),
        }
    }

    fn matches_expected(self) -> bool {
        match self {
            Self::Crc32c {
                expected,
                actual,
            } => actual == expected,
            Self::Sha256 {
                expected,
                actual,
            } => actual.finalize().as_slice() == expected,
            Self::GitBlobSha1 {
                expected,
                actual,
            } => actual.finalize().as_slice() == expected,
        }
    }
}

impl FileFingerprint {
    async fn from_file(file_path: &Path) -> Result<Self, IoError> {
        let file_size = fs::asyn::file_length(file_path).await?;
        let modified = fs::asyn::file_modified(file_path)
            .await?
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|error| IoError::new(ErrorKind::InvalidData, error))?;
        Ok(Self {
            file_size,
            modified_unix_seconds: modified.as_secs(),
            modified_nanos: modified.subsec_nanos(),
        })
    }
}

fn expected_digest_is_well_formed(file_check: &FileCheck) -> bool {
    match file_check {
        FileCheck::CRC(expected) => decode_crc32c(expected).is_some(),
        FileCheck::Sha256(expected) => decode_hex::<32>(expected, "sha256").is_ok(),
        FileCheck::GitBlobSha1(expected) => decode_hex::<20>(expected, "git_blob_sha1").is_ok(),
        FileCheck::None => false,
    }
}

fn decode_crc32c(expected_crc32c_base64: &str) -> Option<u32> {
    let expected_bytes = STANDARD.decode(expected_crc32c_base64).ok()?;
    let expected_bytes: [u8; 4] = expected_bytes.try_into().ok()?;
    Some(u32::from_be_bytes(expected_bytes))
}

fn decode_hex<const LENGTH: usize>(
    encoded: &str,
    algorithm: &'static str,
) -> Result<[u8; LENGTH], VerificationError> {
    let mut decoded = [0; LENGTH];
    decode_to_slice(encoded, &mut decoded).map_err(|_| VerificationError::InvalidExpectedDigest {
        algorithm,
    })?;
    Ok(decoded)
}
