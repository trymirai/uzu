use std::{
    io::{Error as IoError, ErrorKind},
    path::{Path, PathBuf},
};

use base64::Engine;
use kiban::{fs, time::SystemTime};
use serde::{Deserialize, Serialize};
use sha1::Sha1;
use sha2::{Digest, Sha256};

use crate::{FileCheck, backends::common::reject_symlink_components};

const CRC_CACHE_VERSION: u8 = 1;
const DIGEST_CACHE_VERSION: u8 = 1;
const READ_CHUNK_SIZE: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationStatus {
    Match,
    Mismatch,
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
    #[error("invalid expected {algorithm} digest")]
    InvalidExpectedDigest {
        algorithm: &'static str,
    },
    #[error(transparent)]
    Io(#[from] IoError),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
struct CrcCacheReceipt {
    version: u8,
    crc: String,
    file_size: u64,
    modified_unix_seconds: u64,
    modified_nanos: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
struct DigestCacheReceipt {
    version: u8,
    file_check: FileCheck,
    file_size: u64,
    modified_unix_seconds: u64,
    modified_nanos: u32,
}

#[derive(Clone, Copy)]
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
) -> Result<VerificationStatus, VerificationError> {
    if matches!(file_check, FileCheck::None) {
        return Ok(VerificationStatus::Match);
    }

    let file_len = fs::asyn::file_length(file_path).await?;
    let Some(mut verifier) = StreamingVerifier::new(file_check, file_len)? else {
        return Ok(VerificationStatus::Match);
    };

    let bytes_read = stream_file(file_path, |chunk| verifier.update(chunk)).await?;
    if bytes_read != file_len {
        return Err(IoError::new(ErrorKind::UnexpectedEof, "file changed during integrity verification").into());
    }

    if fs::asyn::file_length(file_path).await? != file_len {
        return Err(IoError::new(ErrorKind::UnexpectedEof, "file changed during integrity verification").into());
    }

    Ok(if verifier.matches_expected() {
        VerificationStatus::Match
    } else {
        VerificationStatus::Mismatch
    })
}

#[cfg(not(target_family = "wasm"))]
async fn stream_file(
    file_path: &Path,
    mut consume: impl FnMut(&[u8]),
) -> Result<u64, IoError> {
    use tokio::io::AsyncReadExt;

    let mut file = tokio::fs::File::open(file_path).await?;
    let mut buffer = vec![0_u8; READ_CHUNK_SIZE];
    let mut bytes_read = 0_u64;
    loop {
        let count = file.read(&mut buffer).await?;
        if count == 0 {
            return Ok(bytes_read);
        }
        consume(&buffer[..count]);
        bytes_read = bytes_read.saturating_add(count as u64);
    }
}

#[cfg(target_family = "wasm")]
async fn stream_file(
    file_path: &Path,
    mut consume: impl FnMut(&[u8]),
) -> Result<u64, IoError> {
    let contents = fs::asyn::read(file_path).await?;
    consume(&contents);
    Ok(contents.len() as u64)
}

pub async fn integrity_cache_matches(
    file_path: &Path,
    file_check: &FileCheck,
) -> bool {
    integrity_cache_matches_at(file_path, file_check, &crc_path_for_file(file_path)).await
}

pub(crate) async fn integrity_cache_matches_at(
    file_path: &Path,
    file_check: &FileCheck,
    receipt_path: &Path,
) -> bool {
    if !expected_digest_is_well_formed(file_check) {
        return false;
    }

    let Ok(cache_contents) = fs::asyn::read_to_string(receipt_path).await else {
        return false;
    };
    let Ok(fingerprint) = FileFingerprint::from_file(file_path).await else {
        return false;
    };

    match file_check {
        FileCheck::CRC(crc) => serde_json::from_str::<CrcCacheReceipt>(&cache_contents)
            .is_ok_and(|cached| cached == CrcCacheReceipt::new(crc.clone(), fingerprint)),
        FileCheck::Sha256(_) | FileCheck::GitBlobSha1(_) => serde_json::from_str::<DigestCacheReceipt>(&cache_contents)
            .is_ok_and(|cached| cached == DigestCacheReceipt::new(file_check.clone(), fingerprint)),
        FileCheck::None => false,
    }
}

pub(crate) async fn save_integrity_cache_at(
    file_path: &Path,
    file_check: &FileCheck,
    receipt_path: &Path,
) -> Result<(), IoError> {
    if matches!(file_check, FileCheck::None) {
        return Ok(());
    }

    let fingerprint = FileFingerprint::from_file(file_path).await?;
    let receipt = match file_check {
        FileCheck::CRC(crc) => serde_json::to_vec(&CrcCacheReceipt::new(crc.clone(), fingerprint)),
        FileCheck::Sha256(_) | FileCheck::GitBlobSha1(_) => {
            serde_json::to_vec(&DigestCacheReceipt::new(file_check.clone(), fingerprint))
        },
        FileCheck::None => unreachable!("handled before reading file metadata"),
    }
    .map_err(IoError::other)?;

    if let Some(parent) = receipt_path.parent() {
        fs::asyn::create_dir_all(parent).await?;
    }
    reject_symlink_components(receipt_path).await.map_err(IoError::other)?;
    fs::asyn::write(receipt_path, receipt).await
}

pub fn crc_path_for_file(file_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.crc", file_path.display()))
}

impl StreamingVerifier {
    fn new(
        file_check: &FileCheck,
        file_len: u64,
    ) -> Result<Option<Self>, VerificationError> {
        let verifier = match file_check {
            FileCheck::CRC(expected) => Self::Crc32c {
                expected: decode_crc32c(expected).ok_or(VerificationError::InvalidExpectedDigest {
                    algorithm: "crc32c",
                })?,
                actual: 0,
            },
            FileCheck::Sha256(expected) => Self::Sha256 {
                expected: decode_hex(expected).ok_or(VerificationError::InvalidExpectedDigest {
                    algorithm: "sha256",
                })?,
                actual: Sha256::new(),
            },
            FileCheck::GitBlobSha1(expected) => {
                let mut actual = Sha1::new();
                actual.update(format!("blob {file_len}\0").as_bytes());
                Self::GitBlobSha1 {
                    expected: decode_hex(expected).ok_or(VerificationError::InvalidExpectedDigest {
                        algorithm: "git_blob_sha1",
                    })?,
                    actual,
                }
            },
            FileCheck::None => return Ok(None),
        };
        Ok(Some(verifier))
    }

    fn update(
        &mut self,
        chunk: &[u8],
    ) {
        match self {
            Self::Crc32c {
                actual,
                ..
            } => *actual = crc32c::crc32c_append(*actual, chunk),
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
            } => actual.finalize().as_ref() == expected,
            Self::GitBlobSha1 {
                expected,
                actual,
            } => actual.finalize().as_ref() == expected,
        }
    }
}

impl FileFingerprint {
    async fn from_file(file_path: &Path) -> Result<Self, IoError> {
        let file_size = fs::asyn::file_length(file_path).await?;
        let modified =
            fs::asyn::file_modified(file_path).await?.duration_since(SystemTime::UNIX_EPOCH).map_err(|error| {
                IoError::new(ErrorKind::InvalidData, format!("file modified time predates the Unix epoch: {error}"))
            })?;
        Ok(Self {
            file_size,
            modified_unix_seconds: modified.as_secs(),
            modified_nanos: modified.subsec_nanos(),
        })
    }
}

impl CrcCacheReceipt {
    fn new(
        crc: String,
        fingerprint: FileFingerprint,
    ) -> Self {
        Self {
            version: CRC_CACHE_VERSION,
            crc,
            file_size: fingerprint.file_size,
            modified_unix_seconds: fingerprint.modified_unix_seconds,
            modified_nanos: fingerprint.modified_nanos,
        }
    }
}

impl DigestCacheReceipt {
    fn new(
        file_check: FileCheck,
        fingerprint: FileFingerprint,
    ) -> Self {
        Self {
            version: DIGEST_CACHE_VERSION,
            file_check,
            file_size: fingerprint.file_size,
            modified_unix_seconds: fingerprint.modified_unix_seconds,
            modified_nanos: fingerprint.modified_nanos,
        }
    }
}

fn expected_digest_is_well_formed(file_check: &FileCheck) -> bool {
    match file_check {
        FileCheck::CRC(expected) => decode_crc32c(expected).is_some(),
        FileCheck::Sha256(expected) => decode_hex::<32>(expected).is_some(),
        FileCheck::GitBlobSha1(expected) => decode_hex::<20>(expected).is_some(),
        FileCheck::None => false,
    }
}

fn decode_crc32c(expected_crc32c_base64: &str) -> Option<u32> {
    let expected_bytes = base64::engine::general_purpose::STANDARD.decode(expected_crc32c_base64).ok()?;

    if expected_bytes.len() != 4 {
        return None;
    }

    Some(u32::from_be_bytes([expected_bytes[0], expected_bytes[1], expected_bytes[2], expected_bytes[3]]))
}

fn decode_hex<const LENGTH: usize>(encoded: &str) -> Option<[u8; LENGTH]> {
    if encoded.len() != LENGTH * 2 {
        return None;
    }

    let mut decoded = [0; LENGTH];
    for (index, decoded_byte) in decoded.iter_mut().enumerate() {
        let encoded_index = index * 2;
        *decoded_byte =
            (hex_nibble(encoded.as_bytes()[encoded_index])? << 4) | hex_nibble(encoded.as_bytes()[encoded_index + 1])?;
    }
    Some(decoded)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
