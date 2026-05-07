use std::sync::Arc;

use download_manager::{
    DestinationLockLease, FileCheck, backends::apple::AppleBackendContext, compute_download_id,
    file_download_task_actor::PendingProgressSlot,
    traits::{ActiveDownloadGeneration, BackendContext, BackendEventSender, DownloadConfig},
};
use mock_registry::{Behavior, MockRegistry};
use tokio::{
    runtime::Handle as TokioHandle,
    sync::{Mutex as TokioMutex, mpsc::channel as tokio_mpsc_channel, watch::channel as tokio_watch_channel},
};
use uuid::Uuid;

#[tokio::test(flavor = "multi_thread")]
async fn dropping_apple_active_task_unregisters_event_sink() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temp_dir = tempfile::tempdir()?;
    let destination = temp_dir.path().join(&served_file.file.name);

    let context = AppleBackendContext::new(TokioHandle::current());
    let download_id = compute_download_id(&destination);
    let config = Arc::new(DownloadConfig {
        download_id,
        source_url: served_file.file.url.clone(),
        destination,
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

    let (backend_event_sender, _backend_event_receiver) = tokio_mpsc_channel(64);
    let pending_progress = Arc::new(TokioMutex::new(PendingProgressSlot::default()));
    let (progress_waker_sender, _progress_waker_receiver) = tokio_watch_channel(());
    let backend_event_sender = BackendEventSender::new(backend_event_sender, pending_progress, progress_waker_sender);

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
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temp_dir = tempfile::tempdir()?;
    let destination = temp_dir.path().join(&served_file.file.name);
    let download_id = compute_download_id(&destination);

    let context = AppleBackendContext::new(TokioHandle::current());
    let config = Arc::new(DownloadConfig {
        download_id,
        source_url: served_file.file.url.clone(),
        destination,
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

    let (sender_first, _receiver_first) = tokio_mpsc_channel(64);
    let pending_progress_first = Arc::new(TokioMutex::new(PendingProgressSlot::default()));
    let (waker_first, _waker_recv_first) = tokio_watch_channel(());
    let backend_event_sender_first = BackendEventSender::new(sender_first, pending_progress_first, waker_first);

    let first_task = context
        .download(Arc::clone(&config), ActiveDownloadGeneration::new(0), backend_event_sender_first, &destination_lease)
        .await?;

    let first_keys = context.event_sink_task_identifiers_for_download(download_id);
    assert_eq!(first_keys.len(), 1, "first generation must register a sink");

    let (sender_second, _receiver_second) = tokio_mpsc_channel(64);
    let pending_progress_second = Arc::new(TokioMutex::new(PendingProgressSlot::default()));
    let (waker_second, _waker_recv_second) = tokio_watch_channel(());
    let backend_event_sender_second = BackendEventSender::new(sender_second, pending_progress_second, waker_second);

    let second_task = context
        .download(
            Arc::clone(&config),
            ActiveDownloadGeneration::new(1),
            backend_event_sender_second,
            &destination_lease,
        )
        .await?;

    let both_keys = context.event_sink_task_identifiers_for_download(download_id);
    assert_eq!(
        both_keys.len(),
        2,
        "second generation must not overwrite the first sink - old task callbacks would otherwise pick up the new \
         sink's generation. Got keys: {both_keys:?}",
    );
    assert_ne!(both_keys[0], both_keys[1], "the two generations must have distinct task identifiers");

    drop(first_task);
    drop(second_task);
    destination_lease.release().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn matching_download_task_rejects_persisted_task_with_different_source_url()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temp_dir = tempfile::tempdir()?;
    let destination = temp_dir.path().join(&served_file.file.name);
    let download_id = compute_download_id(&destination);

    let context = AppleBackendContext::new(TokioHandle::current());
    let original_config = Arc::new(DownloadConfig {
        download_id,
        source_url: served_file.file.url.clone(),
        destination: destination.clone(),
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

    let (backend_event_sender, _backend_event_receiver) = tokio_mpsc_channel(64);
    let pending_progress = Arc::new(TokioMutex::new(PendingProgressSlot::default()));
    let (progress_waker_sender, _progress_waker_receiver) = tokio_watch_channel(());
    let backend_event_sender = BackendEventSender::new(backend_event_sender, pending_progress, progress_waker_sender);

    let original_task = context
        .download(
            Arc::clone(&original_config),
            ActiveDownloadGeneration::new(0),
            backend_event_sender,
            &destination_lease,
        )
        .await?;

    let mismatched_config = DownloadConfig {
        source_url: "http://example.invalid/different-url".to_string(),
        ..(*original_config).clone()
    };
    let matched = context.matching_download_task(&mismatched_config).await?;

    assert!(
        matched.is_none(),
        "matching_download_task must refuse to attach a persisted task whose source_url disagrees with the new config",
    );
    drop(original_task);
    destination_lease.release().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn matching_download_task_rejects_persisted_task_with_different_crc() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let temp_dir = tempfile::tempdir()?;
    let destination = temp_dir.path().join(&served_file.file.name);
    let download_id = compute_download_id(&destination);

    let context = AppleBackendContext::new(TokioHandle::current());
    let original_config = Arc::new(DownloadConfig {
        download_id,
        source_url: served_file.file.url.clone(),
        destination: destination.clone(),
        file_check: FileCheck::CRC("00000000".to_string()),
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

    let (backend_event_sender, _backend_event_receiver) = tokio_mpsc_channel(64);
    let pending_progress = Arc::new(TokioMutex::new(PendingProgressSlot::default()));
    let (progress_waker_sender, _progress_waker_receiver) = tokio_watch_channel(());
    let backend_event_sender = BackendEventSender::new(backend_event_sender, pending_progress, progress_waker_sender);

    let original_task = context
        .download(
            Arc::clone(&original_config),
            ActiveDownloadGeneration::new(0),
            backend_event_sender,
            &destination_lease,
        )
        .await?;

    let mismatched_config = DownloadConfig {
        file_check: FileCheck::CRC("ffffffff".to_string()),
        ..(*original_config).clone()
    };
    let matched = context.matching_download_task(&mismatched_config).await?;

    assert!(
        matched.is_none(),
        "matching_download_task must refuse to attach a persisted task whose CRC disagrees with the new config",
    );
    drop(original_task);
    destination_lease.release().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn claim_cancels_mismatched_task() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let served_file = registry.file("tokenizer.json")?;
    let temp_dir = tempfile::tempdir()?;
    let destination = temp_dir.path().join(&served_file.file.name);
    let download_id = compute_download_id(&destination);

    let context = AppleBackendContext::new(TokioHandle::current());
    let original_config = Arc::new(DownloadConfig {
        download_id,
        source_url: served_file.file.url.clone(),
        destination: destination.clone(),
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

    let (backend_event_sender, _backend_event_receiver) = tokio_mpsc_channel(64);
    let pending_progress = Arc::new(TokioMutex::new(PendingProgressSlot::default()));
    let (progress_waker_sender, _progress_waker_receiver) = tokio_watch_channel(());
    let backend_event_sender = BackendEventSender::new(backend_event_sender, pending_progress, progress_waker_sender);

    let original_task = context
        .download(
            Arc::clone(&original_config),
            ActiveDownloadGeneration::new(0),
            backend_event_sender,
            &destination_lease,
        )
        .await?;

    assert!(
        context.matching_download_task(&original_config).await?.is_some(),
        "precondition: original task should be visible before the mismatched claim",
    );

    let mismatched_config = DownloadConfig {
        source_url: "http://example.invalid/different-url".to_string(),
        ..(*original_config).clone()
    };
    assert!(
        context.has_download_task_to_claim(&mismatched_config).await?,
        "manager startup must take the claim path even when a live same-destination URLSession task has mismatched \
         metadata",
    );
    let claimed = context.claim_matching_download_task(&mismatched_config).await?;
    assert!(claimed.is_none(), "mismatched task must not be attached");

    let mut cancelled = false;
    for _attempt in 0..50 {
        if context.matching_download_task(&original_config).await?.is_none() {
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
