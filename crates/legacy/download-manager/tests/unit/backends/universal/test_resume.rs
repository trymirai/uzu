use std::{error::Error, sync::Arc, time::Duration};

use download_manager::{FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase, HttpDownloadRequest};
use kiban::rt::RuntimeHandle;
use tempfile::tempdir;
use tokio::{
    fs::{read as tokio_read, write as tokio_write},
    time::timeout as tokio_timeout,
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};

#[tokio::test(flavor = "multi_thread")]
async fn resume_sends_range_and_restarts_when_ignored() -> Result<(), Box<dyn Error>> {
    let bytes = b"complete file body";
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .and(header("accept-encoding", "identity"))
        .and(header("range", "bytes=7-"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.as_slice()))
        .expect(1)
        .mount(&server)
        .await;

    let temporary_directory = tempdir().unwrap();
    let destination = temporary_directory.path().join("model.bin");
    let part_path = destination.with_added_extension("part");
    tokio_write(&part_path, b"partial").await?;

    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    let task = manager
        .file_download_task(
            (&format!("{}/model.bin", server.uri())).into(),
            &destination,
            FileCheck::None,
            Some(bytes.len() as u64),
        )
        .await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Paused);
    task.download().await?;
    tokio_timeout(Duration::from_secs(10), task.wait()).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    assert_eq!(tokio_read(&destination).await?, bytes);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn authenticated_redirect_strips_authorization_cross_origin() -> Result<(), Box<dyn Error>> {
    let bytes = b"private redirected file";
    let target = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/cdn/model.bin"))
        .and(header("accept-encoding", "identity"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.as_slice()))
        .expect(1)
        .mount(&target)
        .await;
    let origin = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .and(header("authorization", "Bearer test-token"))
        .and(header("accept-encoding", "identity"))
        .respond_with(ResponseTemplate::new(302).insert_header("Location", format!("{}/cdn/model.bin", target.uri())))
        .expect(1)
        .mount(&origin)
        .await;
    let directory = tempdir()?;
    let destination = directory.path().join("model.bin");
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    let token: Arc<str> = Arc::from("test-token");
    let request = HttpDownloadRequest::with_bearer_token(format!("{}/model.bin", origin.uri()), &token);

    let task = manager.file_download_task(request, &destination, FileCheck::None, Some(bytes.len() as u64)).await?;
    task.download().await?;
    tokio_timeout(Duration::from_secs(10), task.wait()).await?;

    let target_requests = target.received_requests().await.expect("requests should be available");
    assert_eq!(target_requests.len(), 1);
    assert!(target_requests[0].headers.get("authorization").is_none());
    assert_eq!(tokio_read(destination).await?, bytes);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn authenticated_resume_sends_bearer_and_range() -> Result<(), Box<dyn Error>> {
    let bytes = b"private file body";
    let resume_from = 7;
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .and(header("authorization", "Bearer test-token"))
        .and(header("range", "bytes=7-"))
        .respond_with(
            ResponseTemplate::new(206)
                .insert_header(
                    "Content-Range",
                    format!("bytes {resume_from}-{}/{}", bytes.len() - 1, bytes.len()),
                )
                .set_body_bytes(&bytes[resume_from..]),
        )
        .expect(1)
        .mount(&server)
        .await;
    let directory = tempdir()?;
    let destination = directory.path().join("model.bin");
    tokio_write(destination.with_added_extension("part"), &bytes[..resume_from]).await?;
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    let token: Arc<str> = Arc::from("test-token");
    let request = HttpDownloadRequest::with_bearer_token(format!("{}/model.bin", server.uri()), &token);
    let task = manager.file_download_task(request, &destination, FileCheck::None, Some(bytes.len() as u64)).await?;

    task.download().await?;
    tokio_timeout(Duration::from_secs(10), task.wait()).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    assert_eq!(tokio_read(destination).await?, bytes);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn signed_redirect_url_is_redacted_from_errors() -> Result<(), Box<dyn Error>> {
    let server = MockServer::start().await;
    Mock::given(method("GET")).and(path("/model.bin")).respond_with(ResponseTemplate::new(500)).mount(&server).await;
    let directory = tempdir()?;
    let destination = directory.path().join("model.bin");
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    let task = manager
        .file_download_task(
            (&format!("{}/model.bin?X-Amz-Signature=super-secret", server.uri())).into(),
            &destination,
            FileCheck::None,
            Some(1),
        )
        .await?;

    task.download().await?;
    tokio_timeout(Duration::from_secs(10), task.wait()).await?;

    let FileDownloadPhase::Error(message) = task.state().await.phase else {
        panic!("request should fail");
    };
    assert!(!message.contains("super-secret"));
    assert!(!message.contains("X-Amz-Signature"));
    Ok(())
}
