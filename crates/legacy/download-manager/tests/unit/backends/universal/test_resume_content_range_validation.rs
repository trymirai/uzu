use std::time::Duration;

use download_manager::{FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase};
use kiban::rt::RuntimeHandle;
use tempfile::tempdir;
use tokio::time::timeout;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

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
    let part_path = destination.with_extension("part");
    tokio::fs::write(&part_path, partial_bytes).await?;

    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    let task = manager
        .file_download_task(&format!("{}/model.bin", server.uri()), &destination, FileCheck::None, Some(total as u64))
        .await?;

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
