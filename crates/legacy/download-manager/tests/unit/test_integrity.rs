use std::error::Error;

use base64::{Engine, engine::general_purpose::STANDARD};
use crc32c::crc32c;
use tempfile::tempdir;
use tokio::fs::write as tokio_write;

use crate::{
    DownloadInfo, FileCheck,
    integrity::{VerificationError, integrity_cache_matches, save_integrity_cache, verify_file_integrity},
};

const HELLO: &[u8] = b"hello\n";
const HELLO_SHA256: &str = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03";
const HELLO_GIT_BLOB_SHA1: &str = "ce013625030ba8dba906f756967f9e9ca394464a";

#[tokio::test]
async fn sha256_detects_same_size_corruption() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let path = directory.path().join("model.bin");
    let check = FileCheck::Sha256(HELLO_SHA256.to_string());
    tokio_write(&path, HELLO).await?;
    assert!(verify_file_integrity(&path, &check).await?);

    tokio_write(&path, b"jello\n").await?;
    assert!(!verify_file_integrity(&path, &check).await?);
    Ok(())
}

#[tokio::test]
async fn git_blob_sha1_includes_the_blob_header() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let path = directory.path().join("config.json");
    let check = FileCheck::GitBlobSha1(HELLO_GIT_BLOB_SHA1.to_string());
    tokio_write(&path, HELLO).await?;

    assert!(verify_file_integrity(&path, &check).await?);
    Ok(())
}

#[tokio::test]
async fn digest_cache_is_invalidated_when_file_changes() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let path = directory.path().join("weights.bin");
    let check = FileCheck::Sha256(HELLO_SHA256.to_string());
    tokio_write(&path, HELLO).await?;
    save_integrity_cache(&path, &check).await?;
    assert!(integrity_cache_matches(&path, &check).await);

    tokio_write(&path, b"jello\n").await?;
    assert!(!integrity_cache_matches(&path, &check).await);
    Ok(())
}

#[tokio::test]
async fn crc_and_invalid_digest_behavior_remain_explicit() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let path = directory.path().join("tokenizer.json");
    tokio_write(&path, HELLO).await?;
    let crc = STANDARD.encode(crc32c(HELLO).to_be_bytes());
    assert!(verify_file_integrity(&path, &FileCheck::CRC(crc)).await?);

    let error = verify_file_integrity(&path, &FileCheck::Sha256("invalid".to_string()))
        .await
        .expect_err("invalid digest must not be treated as corrupt content");
    assert!(matches!(
        error,
        VerificationError::InvalidExpectedDigest {
            algorithm: "sha256"
        }
    ));
    Ok(())
}

#[test]
fn apple_task_info_preserves_new_and_legacy_checks() -> Result<(), Box<dyn Error>> {
    let check = FileCheck::Sha256(HELLO_SHA256.to_string());
    let info = DownloadInfo::new("https://example.com/model", "/tmp/model", check.clone());
    let restored = DownloadInfo::from_json(&info.to_json()?)?;
    assert_eq!(restored.resolved_file_check(), check);

    let legacy = DownloadInfo::from_json(
        r#"{"source_url":"https://example.com/model","destination_path":"/tmp/model","crc32c":"AAAAAA=="}"#,
    )?;
    assert_eq!(legacy.resolved_file_check(), FileCheck::CRC("AAAAAA==".to_string()));
    Ok(())
}
