use std::path::PathBuf;

use uuid::Uuid;

use super::{observe_resume_recovery, write_recovery_metadata};
use crate::{FileCheck, FileState, HttpDownloadRequest, RequestHeaders, compute_download_id, traits::DownloadConfig};

#[tokio::test]
async fn metadata_is_versioned_and_redacts_url_and_authorization() {
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("model.bin");
    let config = test_config(
        destination,
        HttpDownloadRequest::with_headers(
            "https://example.test/private/model.bin?signature=url-secret",
            RequestHeaders::bearer("header-secret").unwrap(),
        ),
        FileCheck::Sha256("a".repeat(64)),
        Some(42),
    );
    tokio::fs::create_dir_all(&config.artifact_root).await.unwrap();
    write_recovery_metadata(&config).await.unwrap();

    let json = tokio::fs::read_to_string(config.recovery_metadata_path()).await.unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["download_id"], config.download_id.to_string());
    assert_eq!(value["expected_bytes"], 42);
    assert!(value["source_fingerprint"].as_str().is_some());
    assert!(!json.contains("example.test"));
    assert!(!json.contains("url-secret"));
    assert!(!json.contains("header-secret"));
    assert!(!json.to_ascii_lowercase().contains("authorization"));
}

#[tokio::test]
async fn changed_source_size_or_integrity_rejects_a_partial_file() {
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("model.bin");
    let original = test_config(
        destination.clone(),
        HttpDownloadRequest::get("https://example.test/model.bin"),
        FileCheck::Sha256("a".repeat(64)),
        Some(42),
    );
    tokio::fs::create_dir_all(&original.artifact_root).await.unwrap();
    let resume_path = original.resume_artifact_path("part");
    tokio::fs::write(&resume_path, b"partial").await.unwrap();
    write_recovery_metadata(&original).await.unwrap();

    let changed_configs = [
        test_config(
            destination.clone(),
            HttpDownloadRequest::get("https://example.test/other.bin"),
            original.file_check.clone(),
            original.expected_bytes,
        ),
        test_config(destination.clone(), original.request.clone(), original.file_check.clone(), Some(43)),
        test_config(destination, original.request.clone(), FileCheck::Sha256("b".repeat(64)), original.expected_bytes),
    ];

    for changed in changed_configs {
        let observation = observe_resume_recovery(&changed, &resume_path).await.unwrap();
        assert_eq!(observation.resume_state, FileState::Missing);
        assert!(observation.cleanup_paths.contains(&resume_path));
        assert!(observation.cleanup_paths.contains(&changed.recovery_metadata_path()));
    }
}

#[tokio::test]
async fn missing_or_unknown_metadata_rejects_a_partial_file() {
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("model.bin");
    let config =
        test_config(destination, HttpDownloadRequest::get("https://example.test/model.bin"), FileCheck::None, Some(42));
    tokio::fs::create_dir_all(&config.artifact_root).await.unwrap();
    let resume_path = config.resume_artifact_path("part");
    tokio::fs::write(&resume_path, b"partial").await.unwrap();

    let missing = observe_resume_recovery(&config, &resume_path).await.unwrap();
    assert_eq!(missing.resume_state, FileState::Missing);
    assert!(missing.cleanup_paths.contains(&resume_path));

    tokio::fs::write(config.recovery_metadata_path(), br#"{"schema_version":99,"destination_path":"model.bin"}"#)
        .await
        .unwrap();
    let unknown = observe_resume_recovery(&config, &resume_path).await.unwrap();
    assert_eq!(unknown.resume_state, FileState::Missing);
    assert!(unknown.cleanup_paths.contains(&config.recovery_metadata_path()));
}

#[cfg(unix)]
#[tokio::test]
async fn metadata_write_rejects_a_late_symlinked_artifact_root() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("model.bin");
    let config =
        test_config(destination, HttpDownloadRequest::get("https://example.test/model.bin"), FileCheck::None, Some(42));
    let outside = directory.path().join("outside");
    tokio::fs::create_dir_all(&outside).await.unwrap();
    tokio::fs::create_dir_all(config.artifact_root.parent().unwrap()).await.unwrap();
    symlink(&outside, &config.artifact_root).unwrap();

    assert!(write_recovery_metadata(&config).await.is_err());
    assert!(!outside.join("recovery.json").exists());
    assert!(!outside.join("recovery.tmp").exists());
}

fn test_config(
    destination: PathBuf,
    request: HttpDownloadRequest,
    file_check: FileCheck,
    expected_bytes: Option<u64>,
) -> DownloadConfig {
    let download_id = compute_download_id(&destination);
    DownloadConfig {
        download_id,
        request,
        artifact_root: DownloadConfig::default_artifact_root(&destination, download_id),
        destination,
        file_check,
        expected_bytes,
        manager_id: "test-manager".to_string(),
        manager_instance_id: Uuid::nil(),
    }
}
