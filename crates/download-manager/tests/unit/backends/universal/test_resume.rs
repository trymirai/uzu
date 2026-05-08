use std::time::Duration;

use download_manager::{FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase};
use tempfile::tempdir;
use tokio::time::timeout;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

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
    let part_path = destination.with_extension("part");
    tokio::fs::write(&part_path, b"partial").await?;

    let manager =
        <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, tokio::runtime::Handle::current()).await?;
    let task = manager
        .file_download_task(&format!("{}/model.bin", server.uri()), &destination, FileCheck::None, Some(bytes.len() as u64))
        .await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Paused);
    task.download().await?;
    timeout(Duration::from_secs(10), task.wait()).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    assert_eq!(tokio::fs::read(&destination).await?, bytes);
    Ok(())
}
