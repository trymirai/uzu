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
async fn resume_rejects_206_when_content_range_offset_mismatches_request() -> Result<(), Box<dyn std::error::Error>> {
    let full_bytes: &[u8] = b"abcdefghij";
    let partial_bytes: &[u8] = b"abcde";
    let total = full_bytes.len();

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .respond_with(
            ResponseTemplate::new(206)
                .set_body_bytes(partial_bytes)
                .insert_header("Content-Range", format!("bytes 0-{}/{}", total - 1, total)),
        )
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
        Some(total as u64),
        partial_bytes,
    )
    .await?;

    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    let task = manager.file_download_task(&source_url, &destination, FileCheck::None, Some(total as u64)).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Paused);
    task.download().await?;
    timeout(Duration::from_secs(10), task.wait()).await?;

    let final_phase = task.state().await.phase;
    match &final_phase {
        FileDownloadPhase::Error(_) => Ok(()),
        FileDownloadPhase::Downloaded => {
            assert_eq!(
                tokio::fs::read(&destination).await?,
                full_bytes,
                "phase reached Downloaded but file content was corrupted by accepting a misaligned Content-Range",
            );
            Ok(())
        },
        other => panic!("unexpected final phase: {:?}", other),
    }
}
