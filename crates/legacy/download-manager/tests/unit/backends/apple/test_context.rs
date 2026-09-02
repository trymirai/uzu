use std::{
    error::Error,
    io::Error as IoError,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use kiban::rt::RuntimeHandle;
use mock_registry::{Behavior, MockRegistry};
use tempfile::tempdir;
use tokio::{
    fs::{read as tokio_read, write as tokio_write},
    sync::{
        Mutex as TokioMutex,
        mpsc::{Receiver as TokioMpscReceiver, channel as tokio_mpsc_channel},
        watch::channel as tokio_watch_channel,
    },
    time::{sleep as tokio_sleep, timeout as tokio_timeout},
};
use uuid::Uuid;
use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{header, method, path},
};

use crate::{
    FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase, HttpDownloadRequest,
    backends::apple::AppleBackendContext,
    common::wait_for_phase,
    compute_download_id,
    file_download_task_actor::{BackendEvent, PendingProgressSlot},
    lock_manager::DestinationLockLease,
    traits::{ActiveDownloadGeneration, ActiveTask, BackendContext, BackendEventSender, DownloadConfig},
};

fn backend_event_sender() -> (BackendEventSender, TokioMpscReceiver<BackendEvent>) {
    let (backend_event_sender, backend_event_receiver) = tokio_mpsc_channel(64);
    let pending_progress = Arc::new(TokioMutex::new(PendingProgressSlot::default()));
    let (progress_waker_sender, _progress_waker_receiver) = tokio_watch_channel(());
    (
        BackendEventSender::new(Uuid::new_v4(), backend_event_sender, pending_progress, progress_waker_sender),
        backend_event_receiver,
    )
}

fn download_config(
    request: impl Into<HttpDownloadRequest>,
    destination: PathBuf,
    expected_bytes: u64,
) -> Arc<DownloadConfig> {
    Arc::new(DownloadConfig {
        download_id: compute_download_id(&destination),
        request: request.into(),
        destination,
        file_check: FileCheck::None,
        expected_bytes: Some(expected_bytes),
        manager_id: "test-manager".to_string(),
        manager_instance_id: Uuid::new_v4(),
    })
}

async fn destination_lease(config: &DownloadConfig) -> Result<DestinationLockLease, IoError> {
    DestinationLockLease::acquire_for_destination(&config.destination, &config.manager_id, config.manager_instance_id)
        .await
}

#[tokio::test(flavor = "multi_thread")]
async fn apple_authenticated_redirect_strips_authorization_cross_origin() -> Result<(), Box<dyn Error>> {
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
    let token: Arc<str> = Arc::from("test-token");
    let config = download_config(
        HttpDownloadRequest::with_bearer_token(format!("{}/model.bin", origin.uri()), &token),
        destination.clone(),
        bytes.len() as u64,
    );
    let lease = destination_lease(&config).await?;
    let generation = ActiveDownloadGeneration::new(0);
    let (sender, mut receiver) = backend_event_sender();
    let context = AppleBackendContext::new(RuntimeHandle::current());

    let active_task = context.download(Arc::clone(&config), generation, sender, &lease).await?;
    let event = tokio_timeout(Duration::from_secs(5), receiver.recv())
        .await?
        .ok_or_else(|| IoError::other("backend event channel closed"))?;

    assert_eq!(event, BackendEvent::completed(generation));
    let target_requests = target.received_requests().await.expect("requests should be available");
    assert_eq!(target_requests.len(), 1);
    assert!(target_requests[0].headers.get("authorization").is_none());
    assert_eq!(tokio_read(destination).await?, bytes);
    drop(active_task);
    lease.release().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn apple_http_error_body_is_not_installed() -> Result<(), Box<dyn Error>> {
    let error_body = b"denied";
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .respond_with(ResponseTemplate::new(404).set_body_bytes(error_body.as_slice()))
        .expect(1)
        .mount(&server)
        .await;

    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join("model.bin");
    let context = AppleBackendContext::new(RuntimeHandle::current());
    let config = download_config(format!("{}/model.bin", server.uri()), destination.clone(), error_body.len() as u64);
    let destination_lease = destination_lease(&config).await?;
    let generation = ActiveDownloadGeneration::new(0);
    let (backend_event_sender, mut backend_event_receiver) = backend_event_sender();

    let active_task =
        context.download(Arc::clone(&config), generation, backend_event_sender, &destination_lease).await?;
    let event = tokio_timeout(Duration::from_secs(5), backend_event_receiver.recv())
        .await?
        .ok_or_else(|| IoError::other("backend event channel closed"))?;

    assert_eq!(event, BackendEvent::error(generation, "download failed with HTTP status 404".to_string()));
    assert!(!destination.exists());

    drop(active_task);
    destination_lease.release().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn apple_download_can_pause_and_resume() -> Result<(), Box<dyn Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let served_file = registry.file("tokenizer.json")?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);

    let context = AppleBackendContext::new(RuntimeHandle::current());
    let config = download_config(served_file.file.url.clone(), destination.clone(), served_file.file.size as u64);
    let destination_lease = destination_lease(&config).await?;
    let initial_generation = ActiveDownloadGeneration::new(0);
    let (initial_event_sender, _initial_event_receiver) = backend_event_sender();
    let active_task =
        context.download(Arc::clone(&config), initial_generation, initial_event_sender, &destination_lease).await?;

    tokio_sleep(Duration::from_millis(100)).await;
    let resume_artifact_path = active_task.pause(&destination).await?;
    assert!(!tokio_read(&resume_artifact_path).await?.is_empty());

    let resumed_generation = ActiveDownloadGeneration::new(1);
    let (resumed_event_sender, mut resumed_event_receiver) = backend_event_sender();
    let resumed_task = context
        .resume(
            Arc::clone(&config),
            resumed_generation,
            &resume_artifact_path,
            resumed_event_sender,
            &destination_lease,
        )
        .await?;
    let event = tokio_timeout(Duration::from_secs(10), resumed_event_receiver.recv())
        .await?
        .ok_or_else(|| IoError::other("backend event channel closed before resumed download completed"))?;

    assert_eq!(event, BackendEvent::completed(resumed_generation));
    assert_eq!(tokio_read(&destination).await?, served_file.bytes.to_vec());

    drop(resumed_task);
    destination_lease.release().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn apple_failed_resume_restarts_from_source() -> Result<(), Box<dyn Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let served_file = registry.file("tokenizer.json")?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Apple, RuntimeHandle::current()).await?;
    let task = manager
        .file_download_task(
            (&served_file.file.url).into(),
            &destination,
            FileCheck::None,
            Some(served_file.file.size as u64),
        )
        .await?;

    task.download().await?;
    tokio_sleep(Duration::from_millis(100)).await;
    task.pause().await?;
    tokio_write(destination.with_added_extension("resume_data"), b"invalid resume data").await?;

    let mut progress = task.progress().await?;
    task.download().await?;
    let state = wait_for_phase(&task, &mut progress, |phase| matches!(phase, FileDownloadPhase::Downloaded)).await;

    assert_eq!(state.downloaded_bytes, served_file.file.size as u64);
    assert_eq!(tokio_read(destination).await?, served_file.bytes.to_vec());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn apple_authenticated_pause_restarts_once_without_resume_data() -> Result<(), Box<dyn Error>> {
    let bytes = b"private model";
    let attempts = Arc::new(AtomicUsize::new(0));
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .and(header("authorization", "Bearer test-token"))
        .respond_with({
            let attempts = Arc::clone(&attempts);
            move |_: &Request| {
                if attempts.fetch_add(1, Ordering::Relaxed) == 0 {
                    ResponseTemplate::new(200).set_body_bytes(bytes.as_slice()).set_delay(Duration::from_secs(1))
                } else {
                    ResponseTemplate::new(500)
                }
            }
        })
        .mount(&server)
        .await;
    let directory = tempdir()?;
    let destination = directory.path().join("model.bin");
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Apple, RuntimeHandle::current()).await?;
    let token: Arc<str> = Arc::from("test-token");
    let request = HttpDownloadRequest::with_bearer_token(format!("{}/model.bin", server.uri()), &token);
    let task = manager.file_download_task(request, &destination, FileCheck::None, Some(bytes.len() as u64)).await?;

    task.download().await?;
    tokio_timeout(Duration::from_secs(5), async {
        while attempts.load(Ordering::Relaxed) == 0 {
            tokio_sleep(Duration::from_millis(10)).await;
        }
    })
    .await?;
    task.pause().await?;

    assert!(tokio_read(destination.with_added_extension("resume_data")).await?.is_empty());
    let mut progress = task.progress().await?;
    task.download().await?;
    wait_for_phase(&task, &mut progress, |phase| matches!(phase, FileDownloadPhase::Error(_))).await;

    assert_eq!(attempts.load(Ordering::Relaxed), 2, "authenticated fresh resume must not be retried");
    assert!(!destination.exists());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn registry_distinguishes_generations_for_same_download_id() -> Result<(), Box<dyn Error>> {
    let bytes = b"model";
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.as_slice()).set_delay(Duration::from_secs(5)))
        .mount(&server)
        .await;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join("model.bin");
    let download_id = compute_download_id(&destination);
    let context = AppleBackendContext::new(RuntimeHandle::current());
    let config = download_config(format!("{}/model.bin", server.uri()), destination, bytes.len() as u64);
    let destination_lease = destination_lease(&config).await?;

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
    assert_eq!(context.event_sink_task_identifiers_for_download(download_id).len(), 1);
    drop(second_task);
    assert!(context.event_sink_task_identifiers_for_download(download_id).is_empty());
    destination_lease.release().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn claim_cancels_task_with_different_credentials() -> Result<(), Box<dyn Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let served_file = registry.file("tokenizer.json")?;
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join(&served_file.file.name);

    let context = AppleBackendContext::new(RuntimeHandle::current());
    let original_token: Arc<str> = Arc::from("original-token");
    let original_config = download_config(
        HttpDownloadRequest::with_bearer_token(served_file.file.url.clone(), &original_token),
        destination,
        served_file.file.size as u64,
    );
    let destination_lease = destination_lease(&original_config).await?;

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

    let replacement_token: Arc<str> = Arc::from("replacement-token");
    let mismatched_config = DownloadConfig {
        request: HttpDownloadRequest::with_bearer_token(served_file.file.url.clone(), &replacement_token),
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
        tokio_sleep(Duration::from_millis(20)).await;
    }

    assert!(
        cancelled,
        "claiming a same-destination task with mismatched metadata must cancel the stale URLSession task",
    );
    drop(original_task);
    destination_lease.release().await?;
    Ok(())
}
