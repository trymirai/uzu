use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use download_manager::{
    DownloadError, FileCheck, FileDownloadGroup, FileDownloadGroupPhase, FileDownloadGroupSpec, FileDownloadManager,
    FileDownloadManagerType, FileDownloadPhase, FileDownloadRequest, HttpDownloadRequest, RelativeFilePath,
    RequestHeaders, compute_download_id, traits::DownloadConfig,
};
use kiban::rt::RuntimeHandle;
use tempfile::tempdir;
use tokio::time::timeout;
use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{header, method, path},
};

use crate::common::write_recoverable_resume_artifact;

async fn universal_manager() -> Result<Box<dyn FileDownloadManager>, download_manager::DownloadError> {
    <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await
}

#[tokio::test(flavor = "multi_thread")]
async fn sends_auth_header_without_exposing_its_value() -> Result<(), Box<dyn std::error::Error>> {
    let token = "hf_secret_token";
    let bytes = b"model bytes";
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .and(header("authorization", format!("Bearer {token}")))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.as_slice()).set_delay(Duration::from_millis(200)))
        .expect(1)
        .mount(&server)
        .await;

    let directory = tempdir()?;
    let destination = directory.path().join("model.bin");
    let manager = universal_manager().await?;
    let request =
        HttpDownloadRequest::with_headers(format!("{}/model.bin", server.uri()), RequestHeaders::bearer(token)?);
    let task =
        manager.http_file_download_task(request, &destination, FileCheck::None, Some(bytes.len() as u64)).await?;

    let task_debug = format!("{task:?}");
    assert!(!task_debug.contains(token), "task Debug leaked the bearer token: {task_debug}");
    task.download().await?;

    let conflicting_request = HttpDownloadRequest::with_headers(
        format!("{}/model.bin", server.uri()),
        RequestHeaders::bearer("different_secret_token")?,
    );
    let conflict = manager
        .http_file_download_task(conflicting_request, &destination, FileCheck::None, Some(bytes.len() as u64))
        .await
        .expect_err("different credentials for the same destination must conflict");
    let conflict_message = conflict.to_string();
    assert!(!conflict_message.contains(token));
    assert!(!conflict_message.contains("different_secret_token"));

    timeout(Duration::from_secs(10), task.wait()).await?;
    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    assert_eq!(tokio::fs::read(destination).await?, bytes);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn strips_authorization_on_cross_origin_redirects() -> Result<(), Box<dyn std::error::Error>> {
    let target = MockServer::start().await;
    let target_saw_authorization = Arc::new(AtomicBool::new(false));
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .respond_with({
            let target_saw_authorization = Arc::clone(&target_saw_authorization);
            move |request: &Request| {
                target_saw_authorization.store(request.headers.contains_key("authorization"), Ordering::SeqCst);
                ResponseTemplate::new(200).set_body_bytes(b"safe".as_slice())
            }
        })
        .expect(1)
        .mount(&target)
        .await;

    let source = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/redirect"))
        .and(header("authorization", "Bearer private-token"))
        .respond_with(ResponseTemplate::new(302).insert_header("Location", format!("{}/model.bin", target.uri())))
        .expect(1)
        .mount(&source)
        .await;

    let directory = tempdir()?;
    let destination = directory.path().join("model.bin");
    let request = HttpDownloadRequest::with_headers(
        format!("{}/redirect", source.uri()),
        RequestHeaders::bearer("private-token")?,
    );
    let task =
        universal_manager().await?.http_file_download_task(request, &destination, FileCheck::None, Some(4)).await?;

    task.download().await?;
    timeout(Duration::from_secs(10), task.wait()).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    assert!(!target_saw_authorization.load(Ordering::SeqCst));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn sends_auth_header_and_managed_range_when_resuming() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .and(header("authorization", "Bearer hf_resume_token"))
        .and(header("range", "bytes=3-"))
        .respond_with(
            ResponseTemplate::new(206).insert_header("Content-Range", "bytes 3-5/6").set_body_bytes(b"def".as_slice()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let directory = tempdir()?;
    let destination = directory.path().join("model.bin");
    let resume_artifact =
        DownloadConfig::resume_artifact_path_for(&destination, compute_download_id(&destination), "part");
    let source_url = format!("{}/model.bin", server.uri());
    write_recoverable_resume_artifact(&resume_artifact, &destination, &source_url, FileCheck::None, Some(6), b"abc")
        .await?;
    let request = HttpDownloadRequest::with_headers(source_url, RequestHeaders::bearer("hf_resume_token")?);
    let task =
        universal_manager().await?.http_file_download_task(request, &destination, FileCheck::None, Some(6)).await?;

    task.download().await?;
    timeout(Duration::from_secs(10), task.wait()).await?;
    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    assert_eq!(tokio::fs::read(destination).await?, b"abcdef");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn retries_server_errors_but_not_auth_errors() -> Result<(), Box<dyn std::error::Error>> {
    let transient_server = MockServer::start().await;
    let transient_attempts = Arc::new(AtomicUsize::new(0));
    Mock::given(method("GET"))
        .and(path("/transient"))
        .respond_with({
            let transient_attempts = Arc::clone(&transient_attempts);
            move |_request: &Request| {
                if transient_attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                    ResponseTemplate::new(500)
                } else {
                    ResponseTemplate::new(200).set_body_bytes(b"ok".as_slice())
                }
            }
        })
        .mount(&transient_server)
        .await;

    let directory = tempdir()?;
    let transient_destination = directory.path().join("transient.bin");
    let transient_task = universal_manager()
        .await?
        .file_download_task(
            &format!("{}/transient", transient_server.uri()),
            &transient_destination,
            FileCheck::None,
            Some(2),
        )
        .await?;
    transient_task.download().await?;
    timeout(Duration::from_secs(10), transient_task.wait()).await?;
    assert_eq!(transient_task.state().await.phase, FileDownloadPhase::Downloaded);
    assert_eq!(transient_attempts.load(Ordering::SeqCst), 2);

    let auth_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/private"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&auth_server)
        .await;
    let auth_destination = directory.path().join("private.bin");
    let auth_task = universal_manager()
        .await?
        .file_download_task(&format!("{}/private", auth_server.uri()), &auth_destination, FileCheck::None, None)
        .await?;
    let auth_snapshots = auth_task.snapshot_receiver();
    auth_task.download().await?;
    timeout(Duration::from_secs(10), auth_task.wait()).await?;
    let snapshot = auth_snapshots.borrow().clone();
    assert!(matches!(snapshot.state.phase, FileDownloadPhase::Error(_)));
    assert_eq!(snapshot.failure, Some(DownloadError::AuthenticationRequired));
    assert_eq!(auth_task.state().await, snapshot.state);
    assert_eq!(auth_task.failure(), snapshot.failure);
    assert!(!auth_destination.exists());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn group_failures_keep_typed_http_status() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/private"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&server)
        .await;

    let directory = tempdir()?;
    let spec = FileDownloadGroupSpec::new(
        directory.path(),
        [FileDownloadRequest::new(
            format!("{}/private", server.uri()),
            RelativeFilePath::try_from("private.bin")?,
            FileCheck::None,
            None,
        )],
    )?;
    let manager: Arc<dyn FileDownloadManager> = Arc::from(universal_manager().await?);
    let group = FileDownloadGroup::open(manager, spec).await?;

    let state = timeout(Duration::from_secs(10), group.download().await?.wait()).await??;

    assert_eq!(state.phase, FileDownloadGroupPhase::Error);
    assert!(matches!(state.failures[0].error, DownloadError::AuthenticationRequired));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn does_not_install_unfollowed_redirect_bodies() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/not-modified"))
        .respond_with(ResponseTemplate::new(304).set_body_bytes(b"not a model".as_slice()))
        .expect(1)
        .mount(&server)
        .await;

    let directory = tempdir()?;
    let destination = directory.path().join("model.bin");
    let task = universal_manager()
        .await?
        .file_download_task(&format!("{}/not-modified", server.uri()), &destination, FileCheck::None, None)
        .await?;

    task.download().await?;
    timeout(Duration::from_secs(10), task.wait()).await?;

    assert!(matches!(task.state().await.phase, FileDownloadPhase::Error(_)));
    assert!(!destination.exists());
    Ok(())
}
