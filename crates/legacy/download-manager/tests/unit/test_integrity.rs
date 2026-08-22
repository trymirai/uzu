use base64::Engine;

use crate::{
    DownloadError, DownloadInfo, FileCheck, FileState, compute_download_id,
    crc_utils::{
        VerificationError, VerificationStatus, crc_path_for_file, integrity_cache_matches, save_integrity_cache_at,
        verify_file_integrity,
    },
    reducer::{DiskObservation, validate},
    traits::DownloadConfig,
};

const HELLO: &[u8] = b"hello\n";
const HELLO_SHA256: &str = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03";
const HELLO_GIT_BLOB_SHA1: &str = "ce013625030ba8dba906f756967f9e9ca394464a";

async fn save_integrity_cache(
    file_path: &std::path::Path,
    file_check: &FileCheck,
) -> Result<(), std::io::Error> {
    save_integrity_cache_at(file_path, file_check, &crc_path_for_file(file_path)).await
}

#[test]
fn apple_download_info_preserves_legacy_crc_and_new_digest_checks() -> Result<(), Box<dyn std::error::Error>> {
    let legacy_json =
        r#"{"source_url":"https://example.test/file","destination_path":"/tmp/file","crc32c":"AAAAAA=="}"#;
    let legacy = DownloadInfo::from_json(legacy_json)?;
    assert_eq!(legacy.resolved_file_check(), FileCheck::CRC("AAAAAA==".to_string()));

    let sha = DownloadInfo::with_file_check(
        "https://example.test/file",
        "/tmp/file",
        FileCheck::Sha256(HELLO_SHA256.to_string()),
    );
    assert_eq!(
        DownloadInfo::from_json(&sha.to_json()?)?.resolved_file_check(),
        FileCheck::Sha256(HELLO_SHA256.to_string())
    );
    Ok(())
}

#[tokio::test]
async fn sha256_detects_valid_corrupt_and_truncated_files() -> Result<(), Box<dyn std::error::Error>> {
    let temporary_directory = tempfile::tempdir()?;
    let file_path = temporary_directory.path().join("model.bin");
    let file_check = FileCheck::Sha256(HELLO_SHA256.to_string());

    tokio::fs::write(&file_path, HELLO).await?;
    assert_eq!(verify_file_integrity(&file_path, &file_check).await?, VerificationStatus::Match);

    tokio::fs::write(&file_path, b"jello\n").await?;
    assert_eq!(verify_file_integrity(&file_path, &file_check).await?, VerificationStatus::Mismatch);

    tokio::fs::write(&file_path, b"hello").await?;
    assert_eq!(verify_file_integrity(&file_path, &file_check).await?, VerificationStatus::Mismatch);
    Ok(())
}

#[tokio::test]
async fn git_blob_sha1_hashes_the_blob_header_and_contents() -> Result<(), Box<dyn std::error::Error>> {
    let temporary_directory = tempfile::tempdir()?;
    let file_path = temporary_directory.path().join("config.json");
    let file_check = FileCheck::GitBlobSha1(HELLO_GIT_BLOB_SHA1.to_string());

    tokio::fs::write(&file_path, HELLO).await?;
    assert_eq!(verify_file_integrity(&file_path, &file_check).await?, VerificationStatus::Match);

    tokio::fs::write(&file_path, b"jello\n").await?;
    assert_eq!(verify_file_integrity(&file_path, &file_check).await?, VerificationStatus::Mismatch);

    tokio::fs::write(&file_path, b"hello").await?;
    assert_eq!(verify_file_integrity(&file_path, &file_check).await?, VerificationStatus::Mismatch);
    Ok(())
}

#[tokio::test]
async fn crc32c_and_none_checks_remain_compatible() -> Result<(), Box<dyn std::error::Error>> {
    let temporary_directory = tempfile::tempdir()?;
    let file_path = temporary_directory.path().join("tokenizer.json");
    tokio::fs::write(&file_path, HELLO).await?;

    let crc = crc32c::crc32c(HELLO);
    let encoded_crc = base64::engine::general_purpose::STANDARD.encode(crc.to_be_bytes());
    assert_eq!(verify_file_integrity(&file_path, &FileCheck::CRC(encoded_crc)).await?, VerificationStatus::Match);

    let missing_path = temporary_directory.path().join("missing");
    assert_eq!(verify_file_integrity(&missing_path, &FileCheck::None).await?, VerificationStatus::Match);
    Ok(())
}

#[tokio::test]
async fn sha_cache_is_invalidated_when_the_file_changes() -> Result<(), Box<dyn std::error::Error>> {
    let temporary_directory = tempfile::tempdir()?;
    let file_path = temporary_directory.path().join("weights.bin");
    let file_check = FileCheck::Sha256(HELLO_SHA256.to_string());
    tokio::fs::write(&file_path, HELLO).await?;

    save_integrity_cache(&file_path, &file_check).await?;
    assert!(integrity_cache_matches(&file_path, &file_check).await);

    tokio::fs::write(&file_path, b"jello\n").await?;
    assert!(!integrity_cache_matches(&file_path, &file_check).await);
    Ok(())
}

#[tokio::test]
async fn invalid_digest_is_not_reported_as_corrupt_content() -> Result<(), Box<dyn std::error::Error>> {
    let temporary_directory = tempfile::tempdir()?;
    let file_path = temporary_directory.path().join("model.bin");
    tokio::fs::write(&file_path, HELLO).await?;

    let error = verify_file_integrity(&file_path, &FileCheck::Sha256("not-a-digest".to_string()))
        .await
        .expect_err("malformed expected digest must be a configuration error");
    assert!(matches!(
        error,
        VerificationError::InvalidExpectedDigest {
            algorithm: "sha256"
        }
    ));
    assert_eq!(tokio::fs::read(&file_path).await?, HELLO);
    Ok(())
}

#[tokio::test]
async fn verification_io_failure_aborts_startup_without_a_delete_plan() -> Result<(), Box<dyn std::error::Error>> {
    let temporary_directory = tempfile::tempdir()?;
    let destination_path = temporary_directory.path().join("model.bin");
    tokio::fs::write(&destination_path, HELLO).await?;
    tokio::fs::remove_file(&destination_path).await?;

    let observation = DiskObservation {
        destination_state: FileState::Exists,
        crc_state: FileState::Missing,
        resume_state: FileState::Missing,
        destination_size: Some(HELLO.len() as u64),
        resume_size: None,
        file_check: FileCheck::Sha256(HELLO_SHA256.to_string()),
        expected_bytes: Some(HELLO.len() as u64),
        destination_path: destination_path.clone(),
        crc_path: Some(destination_path.with_extension("crc")),
        resume_artifact_path: Some(DownloadConfig::resume_artifact_path_for(
            &destination_path,
            compute_download_id(&destination_path),
            "part",
        )),
    };

    let error = validate(&observation).await.expect_err("a read race must abort validation");
    assert!(matches!(error, DownloadError::IntegrityIo { .. }));
    assert!(!destination_path.exists());
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
async fn unreadable_destination_is_an_io_failure_and_is_preserved() -> Result<(), Box<dyn std::error::Error>> {
    use std::os::unix::fs::PermissionsExt;

    let temporary_directory = tempfile::tempdir()?;
    let destination_path = temporary_directory.path().join("model.bin");
    tokio::fs::write(&destination_path, HELLO).await?;
    let original_permissions = std::fs::metadata(&destination_path)?.permissions();
    std::fs::set_permissions(&destination_path, std::fs::Permissions::from_mode(0o000))?;

    if tokio::fs::File::open(&destination_path).await.is_ok() {
        std::fs::set_permissions(&destination_path, original_permissions)?;
        return Ok(());
    }

    let observation = DiskObservation {
        destination_state: FileState::Exists,
        crc_state: FileState::Missing,
        resume_state: FileState::Missing,
        destination_size: Some(HELLO.len() as u64),
        resume_size: None,
        file_check: FileCheck::Sha256(HELLO_SHA256.to_string()),
        expected_bytes: Some(HELLO.len() as u64),
        destination_path: destination_path.clone(),
        crc_path: Some(destination_path.with_extension("crc")),
        resume_artifact_path: Some(DownloadConfig::resume_artifact_path_for(
            &destination_path,
            compute_download_id(&destination_path),
            "part",
        )),
    };

    let validation = validate(&observation).await;
    let destination_still_exists = std::fs::symlink_metadata(&destination_path)?.is_file();
    std::fs::set_permissions(&destination_path, original_permissions)?;

    assert!(matches!(validation, Err(DownloadError::IntegrityIo { .. })));
    assert!(destination_still_exists);
    assert_eq!(tokio::fs::read(&destination_path).await?, HELLO);
    Ok(())
}
