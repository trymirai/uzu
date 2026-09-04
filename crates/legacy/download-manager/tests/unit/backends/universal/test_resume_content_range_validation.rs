use std::{error::Error, time::Duration};

use download_manager::{FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase};
use kiban::rt::RuntimeHandle;
use rstest::rstest;
use tempfile::tempdir;
use tokio::{
    fs::{read as tokio_read, write as tokio_write},
    time::timeout as tokio_timeout,
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};

#[rstest]
#[case::offset("bytes 0-4/10")]
#[case::total("bytes 5-9/11")]
#[tokio::test(flavor = "multi_thread")]
async fn invalid_content_range_restarts_from_zero(#[case] content_range: &str) -> Result<(), Box<dyn Error>> {
    let full_bytes: &[u8] = b"abcdefghij";
    let partial_bytes: &[u8] = b"abcde";
    let total = full_bytes.len();

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .and(header("range", "bytes=5-"))
        .respond_with(
            ResponseTemplate::new(206)
                .set_body_bytes(&full_bytes[partial_bytes.len()..])
                .insert_header("Content-Range", content_range),
        )
        .with_priority(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(full_bytes))
        .with_priority(2)
        .expect(1)
        .mount(&server)
        .await;

    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join("model.bin");
    let part_path = destination.with_added_extension("part");
    tokio_write(&part_path, partial_bytes).await?;

    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    let task = manager
        .file_download_task(
            (&format!("{}/model.bin", server.uri())).into(),
            &destination,
            FileCheck::None,
            Some(total as u64),
        )
        .await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Paused);
    task.download().await?;
    tokio_timeout(Duration::from_secs(10), task.wait()).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    assert_eq!(tokio_read(destination).await?, full_bytes);
    Ok(())
}
