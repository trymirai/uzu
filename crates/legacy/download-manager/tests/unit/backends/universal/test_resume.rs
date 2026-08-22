use std::time::Duration;

use download_manager::{
    FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase, compute_download_id,
    traits::DownloadConfig,
};
use kiban::rt::RuntimeHandle;
use tempfile::tempdir;
use tokio::time::timeout;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

use crate::common::write_recoverable_resume_artifact;

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_resume_restarts_when_server_ignores_range() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = b"complete file body";
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.as_slice()))
        .mount(&server)
        .await;

    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join("model.bin");
    let part_path = DownloadConfig::resume_artifact_path_for(&destination, compute_download_id(&destination), "part");
    let source_url = format!("{}/model.bin", server.uri());
    write_recoverable_resume_artifact(
        &part_path,
        &destination,
        &source_url,
        FileCheck::None,
        Some(bytes.len() as u64),
        b"partial",
    )
    .await?;

    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    let task = manager.file_download_task(&source_url, &destination, FileCheck::None, Some(bytes.len() as u64)).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Paused);
    task.download().await?;
    timeout(Duration::from_secs(10), task.wait()).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    assert_eq!(tokio::fs::read(&destination).await?, bytes);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn restart_discards_a_partial_file_when_the_source_changes() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = b"new immutable model";
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/new-model.bin"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.as_slice()))
        .expect(1)
        .mount(&server)
        .await;

    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join("model.bin");
    let part_path = DownloadConfig::resume_artifact_path_for(&destination, compute_download_id(&destination), "part");
    write_recoverable_resume_artifact(
        &part_path,
        &destination,
        "https://old.example.test/model.bin",
        FileCheck::None,
        Some(bytes.len() as u64),
        b"old partial bytes",
    )
    .await?;
    let metadata_path = part_path.parent().expect("part path must have a parent").join("recovery.json");

    let source_url = format!("{}/new-model.bin", server.uri());
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    let task = manager.file_download_task(&source_url, &destination, FileCheck::None, Some(bytes.len() as u64)).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::NotDownloaded);
    assert!(!part_path.exists(), "startup must discard a partial file owned by the old source");
    assert!(!metadata_path.exists(), "startup must discard metadata owned by the old source");

    task.download().await?;
    timeout(Duration::from_secs(10), task.wait()).await?;
    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    assert_eq!(tokio::fs::read(&destination).await?, bytes);
    let requests = server.received_requests().await.expect("request recording is enabled");
    assert_eq!(requests.len(), 1);
    assert!(!requests[0].headers.contains_key("range"), "the new source must start from byte zero");
    Ok(())
}
