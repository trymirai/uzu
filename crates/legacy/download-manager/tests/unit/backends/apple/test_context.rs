use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use kiban::rt::RuntimeHandle;
use mock_registry::{Behavior, MockRegistry};
use tokio::sync::{
    Mutex as TokioMutex,
    mpsc::{Receiver as TokioMpscReceiver, channel as tokio_mpsc_channel},
    watch::channel as tokio_watch_channel,
};
use uuid::Uuid;
use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{header, method, path},
};

use crate::{
    FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase, HttpDownloadRequest, RequestHeaders,
    backends::apple::AppleBackendContext,
    compute_download_id,
    file_download_task_actor::BackendEvent,
    lock_manager::DestinationLockLease,
    recovery_metadata::write_recovery_metadata,
    traits::{ActiveDownloadGeneration, BackendContext, BackendEventSender, DownloadConfig},
};

fn backend_event_sender() -> (BackendEventSender, TokioMpscReceiver<BackendEvent>) {
    let (backend_event_sender, backend_event_receiver) = tokio_mpsc_channel(64);
    let pending_progress = Arc::new(TokioMutex::new(None));
    let (progress_waker_sender, _progress_waker_receiver) = tokio_watch_channel(());
    (
        BackendEventSender::new(Uuid::new_v4(), backend_event_sender, pending_progress, progress_waker_sender),
        backend_event_receiver,
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_resume_empty_resume_data_starts_fresh_download() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temporary_directory = tempfile::tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let resume_artifact_path =
        DownloadConfig::resume_artifact_path_for(&destination, compute_download_id(&destination), "resume_data");
    let artifact_root = resume_artifact_path.parent().expect("resume artifact has a parent").to_path_buf();
    tokio::fs::create_dir_all(&artifact_root).await?;
    tokio::fs::write(&resume_artifact_path, b"").await?;

    let context = AppleBackendContext::new(RuntimeHandle::current());
    let config = Arc::new(DownloadConfig {
        download_id: compute_download_id(&destination),
        request: HttpDownloadRequest::get(served_file.file.url.clone()),
        destination: destination.clone(),
        artifact_root,
        file_check: FileCheck::None,
        expected_bytes: Some(served_file.file.size as u64),
        manager_id: "test-manager".to_string(),
        manager_instance_id: Uuid::new_v4(),
    });
    write_recovery_metadata(&config).await?;
    let destination_lease = DestinationLockLease::acquire_for_destination(
        &config.destination,
        &config.manager_id,
        config.manager_instance_id,
    )
    .await?;
    let generation = ActiveDownloadGeneration::new(0);
    let (backend_event_sender, mut backend_event_receiver) = backend_event_sender();

    let active_task = context
        .resume(Arc::clone(&config), generation, &resume_artifact_path, backend_event_sender, &destination_lease)
        .await?;
    let event = tokio::time::timeout(std::time::Duration::from_secs(5), backend_event_receiver.recv())
        .await?
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "backend event channel closed before empty resume-data fallback completed",
            )
        })?;

    assert_eq!(event, BackendEvent::completed(generation));
    assert_eq!(tokio::fs::read(&destination).await?, served_file.bytes.to_vec());

    drop(active_task);
    destination_lease.release().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn authenticated_pause_writes_only_an_empty_restart_marker() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let served_file = registry.file("tokenizer.json")?;
    let temporary_directory = tempfile::tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let resume_artifact_path =
        DownloadConfig::resume_artifact_path_for(&destination, compute_download_id(&destination), "resume_data");
    let request =
        HttpDownloadRequest::with_headers(served_file.file.url.clone(), RequestHeaders::bearer("pause_secret_token")?);
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Apple, RuntimeHandle::current()).await?;
    let task = manager
        .http_file_download_task(request, &destination, FileCheck::None, Some(served_file.file.size as u64))
        .await?;

    task.download().await?;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let state = task.state().await;
            if matches!(state.phase, FileDownloadPhase::Downloading) && state.downloaded_bytes > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await?;
    task.pause().await?;

    assert_eq!(tokio::fs::read(&resume_artifact_path).await?, b"");
    assert!(matches!(task.state().await.phase, FileDownloadPhase::Paused));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn unauthenticated_pause_preserves_opaque_resume_data() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let served_file = registry.file("tokenizer.json")?;
    let temporary_directory = tempfile::tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let resume_artifact_path =
        DownloadConfig::resume_artifact_path_for(&destination, compute_download_id(&destination), "resume_data");
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Apple, RuntimeHandle::current()).await?;
    let task = manager
        .file_download_task(&served_file.file.url, &destination, FileCheck::None, Some(served_file.file.size as u64))
        .await?;

    task.download().await?;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let state = task.state().await;
            if matches!(state.phase, FileDownloadPhase::Downloading) && state.downloaded_bytes > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await?;
    task.pause().await?;

    assert!(!tokio::fs::read(&resume_artifact_path).await?.is_empty());
    assert!(matches!(task.state().await.phase, FileDownloadPhase::Paused));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn dropping_apple_active_task_unregisters_event_sink() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let served_file = registry.file("config.json")?;
    let temporary_directory = tempfile::tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);

    let context = AppleBackendContext::new(RuntimeHandle::current());
    let download_id = compute_download_id(&destination);
    let config = Arc::new(DownloadConfig {
        download_id,
        request: HttpDownloadRequest::get(served_file.file.url.clone()),
        destination,
        artifact_root: temporary_directory.path().join("artifacts"),
        file_check: FileCheck::None,
        expected_bytes: Some(served_file.file.size as u64),
        manager_id: "test-manager".to_string(),
        manager_instance_id: Uuid::new_v4(),
    });
    let destination_lease = DestinationLockLease::acquire_for_destination(
        &config.destination,
        &config.manager_id,
        config.manager_instance_id,
    )
    .await?;

    let (backend_event_sender, _backend_event_receiver) = backend_event_sender();
    let active_task = context
        .download(Arc::clone(&config), ActiveDownloadGeneration::new(0), backend_event_sender, &destination_lease)
        .await?;
    assert_eq!(
        context.event_sink_count_for_download(download_id),
        1,
        "precondition: exactly one event sink should be registered for download_id after starting download",
    );

    drop(active_task);
    destination_lease.release().await?;

    assert_eq!(
        context.event_sink_count_for_download(download_id),
        0,
        "dropping AppleActiveTask must unregister its event sink",
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn registry_distinguishes_generations_for_same_download_id() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    // Keep both native tasks alive long enough to observe their registrations.
    // `config.json` fits in one throttled chunk and can finish before the second
    // task is registered on a busy full-suite run.
    let served_file = registry.file("tokenizer.json")?;
    let temporary_directory = tempfile::tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let download_id = compute_download_id(&destination);

    let context = AppleBackendContext::new(RuntimeHandle::current());
    let config = Arc::new(DownloadConfig {
        download_id,
        request: HttpDownloadRequest::get(served_file.file.url.clone()),
        destination,
        artifact_root: temporary_directory.path().join("artifacts"),
        file_check: FileCheck::None,
        expected_bytes: Some(served_file.file.size as u64),
        manager_id: "test-manager".to_string(),
        manager_instance_id: Uuid::new_v4(),
    });
    let destination_lease = DestinationLockLease::acquire_for_destination(
        &config.destination,
        &config.manager_id,
        config.manager_instance_id,
    )
    .await?;

    let (backend_event_sender_first, _backend_event_receiver_first) = backend_event_sender();
    let first_task = context
        .download(Arc::clone(&config), ActiveDownloadGeneration::new(0), backend_event_sender_first, &destination_lease)
        .await?;
    assert_eq!(
        context.event_sink_task_identifiers_for_download(download_id).len(),
        1,
        "first generation must register a sink",
    );

    let (backend_event_sender_second, _backend_event_receiver_second) = backend_event_sender();
    let second_task = context
        .download(
            Arc::clone(&config),
            ActiveDownloadGeneration::new(1),
            backend_event_sender_second,
            &destination_lease,
        )
        .await?;

    let both_keys = context.event_sink_task_identifiers_for_download(download_id);
    assert_eq!(both_keys.len(), 2, "second generation must not overwrite the first sink; got keys: {both_keys:?}");
    assert_ne!(both_keys[0], both_keys[1], "the two generations must have distinct task identifiers");

    drop(first_task);
    drop(second_task);
    destination_lease.release().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn claim_cancels_mismatched_task() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let served_file = registry.file("tokenizer.json")?;
    let temporary_directory = tempfile::tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let download_id = compute_download_id(&destination);

    let context = AppleBackendContext::new(RuntimeHandle::current());
    let original_config = Arc::new(DownloadConfig {
        download_id,
        request: HttpDownloadRequest::get(served_file.file.url.clone()),
        destination: destination.clone(),
        artifact_root: temporary_directory.path().join("artifacts"),
        file_check: FileCheck::None,
        expected_bytes: Some(served_file.file.size as u64),
        manager_id: "test-manager".to_string(),
        manager_instance_id: Uuid::new_v4(),
    });
    let destination_lease = DestinationLockLease::acquire_for_destination(
        &original_config.destination,
        &original_config.manager_id,
        original_config.manager_instance_id,
    )
    .await?;

    let (backend_event_sender, _backend_event_receiver) = backend_event_sender();
    let original_task = context
        .download(
            Arc::clone(&original_config),
            ActiveDownloadGeneration::new(0),
            backend_event_sender,
            &destination_lease,
        )
        .await?;

    assert!(
        context.claim_matching_download_task(&original_config).await?.is_some(),
        "precondition: original task should be visible before the mismatched claim",
    );

    let mismatched_config = DownloadConfig {
        request: HttpDownloadRequest::get("http://example.invalid/different-url"),
        ..(*original_config).clone()
    };
    assert!(
        context.has_download_task_to_claim(&mismatched_config).await?,
        "manager startup must take the claim path for a live same-destination task with mismatched metadata",
    );
    assert!(context.claim_matching_download_task(&mismatched_config).await?.is_none());

    let mut cancelled = false;
    for _attempt in 0..50 {
        if context.claim_matching_download_task(&original_config).await?.is_none() {
            cancelled = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    assert!(
        cancelled,
        "claiming a same-destination task with mismatched metadata must cancel the stale URLSession task",
    );
    drop(original_task);
    destination_lease.release().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn authenticated_claim_cancels_a_legacy_recovery_task() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let served_file = registry.file("tokenizer.json")?;
    let temporary_directory = tempfile::tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let download_id = compute_download_id(&destination);

    let context = AppleBackendContext::new(RuntimeHandle::current());
    let public_config = Arc::new(DownloadConfig {
        download_id,
        request: HttpDownloadRequest::get(served_file.file.url.clone()),
        destination: destination.clone(),
        artifact_root: temporary_directory.path().join("artifacts"),
        file_check: FileCheck::None,
        expected_bytes: Some(served_file.file.size as u64),
        manager_id: "test-manager".to_string(),
        manager_instance_id: Uuid::new_v4(),
    });
    let destination_lease = DestinationLockLease::acquire_for_destination(
        &public_config.destination,
        &public_config.manager_id,
        public_config.manager_instance_id,
    )
    .await?;
    let (backend_event_sender, _backend_event_receiver) = backend_event_sender();
    let public_task = context
        .download(
            Arc::clone(&public_config),
            ActiveDownloadGeneration::new(0),
            backend_event_sender,
            &destination_lease,
        )
        .await?;

    let authenticated_config = DownloadConfig {
        request: HttpDownloadRequest::with_headers(
            served_file.file.url.clone(),
            RequestHeaders::bearer("legacy_task_token")?,
        ),
        ..(*public_config).clone()
    };
    assert!(
        !context.has_download_task_to_claim(&authenticated_config).await?,
        "authenticated requests must not attach to recoverable session tasks",
    );
    assert!(context.claim_matching_download_task(&authenticated_config).await?.is_none());
    assert!(context.claim_matching_download_task(&public_config).await?.is_none());

    drop(public_task);
    destination_lease.release().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn apple_sends_auth_header_and_does_not_install_http_errors() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/authorized"))
        .and(header("authorization", "Bearer apple_secret_token"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"ok".as_slice()))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/missing"))
        .respond_with(ResponseTemplate::new(404).set_body_bytes(b"not a model".as_slice()))
        .expect(1)
        .mount(&server)
        .await;

    let directory = tempfile::tempdir()?;
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Apple, RuntimeHandle::current()).await?;
    let authorized_destination = directory.path().join("authorized.bin");
    let authorized_request = HttpDownloadRequest::with_headers(
        format!("{}/authorized", server.uri()),
        RequestHeaders::bearer("apple_secret_token")?,
    );
    let authorized_task =
        manager.http_file_download_task(authorized_request, &authorized_destination, FileCheck::None, Some(2)).await?;
    authorized_task.download().await?;
    tokio::time::timeout(Duration::from_secs(10), authorized_task.wait()).await?;
    assert_eq!(authorized_task.state().await.phase, FileDownloadPhase::Downloaded);
    assert_eq!(tokio::fs::read(authorized_destination).await?, b"ok");

    let missing_destination = directory.path().join("missing.bin");
    let missing_task = manager
        .file_download_task(&format!("{}/missing", server.uri()), &missing_destination, FileCheck::None, None)
        .await?;
    missing_task.download().await?;
    tokio::time::timeout(Duration::from_secs(10), missing_task.wait()).await?;
    assert!(matches!(missing_task.state().await.phase, FileDownloadPhase::Error(_)));
    assert!(!missing_destination.exists(), "HTTP error body must not be installed as the destination");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn apple_retries_a_transient_http_error() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    let attempts = Arc::new(AtomicUsize::new(0));
    Mock::given(method("GET"))
        .and(path("/transient"))
        .respond_with({
            let attempts = Arc::clone(&attempts);
            move |_request: &Request| {
                if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                    ResponseTemplate::new(500)
                } else {
                    ResponseTemplate::new(200).set_body_bytes(b"ok".as_slice())
                }
            }
        })
        .mount(&server)
        .await;

    let directory = tempfile::tempdir()?;
    let destination = directory.path().join("transient.bin");
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Apple, RuntimeHandle::current()).await?;
    let task = manager
        .file_download_task(&format!("{}/transient", server.uri()), &destination, FileCheck::None, Some(2))
        .await?;

    task.download().await?;
    tokio::time::timeout(Duration::from_secs(10), task.wait()).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert_eq!(tokio::fs::read(destination).await?, b"ok");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn apple_does_not_forward_authorization_to_redirected_origin() -> Result<(), Box<dyn std::error::Error>> {
    let redirected_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"model".as_slice()))
        .expect(1)
        .mount(&redirected_server)
        .await;

    let source_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/resolve/model.bin"))
        .and(header("authorization", "Bearer apple_redirect_token"))
        .respond_with(
            ResponseTemplate::new(302).insert_header("Location", format!("{}/model.bin", redirected_server.uri())),
        )
        .expect(1)
        .mount(&source_server)
        .await;

    let directory = tempfile::tempdir()?;
    let destination = directory.path().join("model.bin");
    let request = HttpDownloadRequest::with_headers(
        format!("{}/resolve/model.bin", source_server.uri()),
        RequestHeaders::bearer("apple_redirect_token")?,
    );
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Apple, RuntimeHandle::current()).await?;
    let task = manager.http_file_download_task(request, &destination, FileCheck::None, Some(5)).await?;

    task.download().await?;
    tokio::time::timeout(Duration::from_secs(10), task.wait()).await?;

    assert_eq!(task.state().await.phase, FileDownloadPhase::Downloaded);
    assert_eq!(tokio::fs::read(destination).await?, b"model");
    let redirected_requests = redirected_server.received_requests().await.expect("request recording is enabled");
    assert_eq!(redirected_requests.len(), 1);
    assert!(
        !redirected_requests[0].headers.contains_key("authorization"),
        "Authorization must be stripped before following a cross-origin redirect",
    );
    Ok(())
}
