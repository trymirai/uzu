use std::{error::Error, sync::Arc, time::Duration};

use kiban::{fs, rt::RuntimeHandle};
use tempfile::tempdir;
use tokio::{
    sync::{
        Mutex as TokioMutex,
        mpsc::{Receiver as TokioMpscReceiver, channel as tokio_mpsc_channel},
        watch::channel as tokio_watch_channel,
    },
    time::timeout as tokio_timeout,
};
use uuid::Uuid;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

use crate::{
    FileCheck,
    backends::universal::UniversalBackendContext,
    compute_download_id,
    file_download_task_actor::{BackendEvent, PendingProgressSlot},
    lock_manager::DestinationLockLease,
    traits::{ActiveDownloadGeneration, BackendContext, BackendEventSender, DownloadConfig},
};

fn backend_event_sender() -> (BackendEventSender, TokioMpscReceiver<BackendEvent>) {
    let (sender, receiver) = tokio_mpsc_channel(8);
    let pending_progress = Arc::new(TokioMutex::new(PendingProgressSlot::default()));
    let (progress_waker, _) = tokio_watch_channel(());
    (BackendEventSender::new(Uuid::new_v4(), sender, pending_progress, progress_waker), receiver)
}

#[tokio::test(flavor = "multi_thread")]
async fn response_body_cannot_grow_part_beyond_expected_size() -> Result<(), Box<dyn Error>> {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![0; 256 * 1024]))
        .expect(1)
        .mount(&server)
        .await;
    let directory = tempdir()?;
    let destination = directory.path().join("model.bin");
    let config = Arc::new(DownloadConfig {
        download_id: compute_download_id(&destination),
        request: server.uri().into(),
        destination: destination.clone(),
        file_check: FileCheck::None,
        expected_bytes: Some(1),
        manager_id: "test-manager".to_string(),
        manager_instance_id: Uuid::new_v4(),
    });
    let destination_lease = DestinationLockLease::acquire_for_destination(
        &destination,
        &config.manager_id,
        config.manager_instance_id,
    )
    .await?;
    let mut context = UniversalBackendContext::new(RuntimeHandle::current());
    context.retries = 0;
    let (sender, mut receiver) = backend_event_sender();

    let active_task = context
        .download(
            Arc::clone(&config),
            ActiveDownloadGeneration::new(0),
            sender,
            &destination_lease,
        )
        .await?;
    let event = tokio_timeout(Duration::from_secs(5), receiver.recv()).await?;

    assert!(matches!(event, Some(BackendEvent::Error { .. })));
    assert!(fs::asyn::file_length(destination.with_added_extension("part")).await.unwrap_or(0) <= 1);
    assert!(!destination.exists());
    drop(active_task);
    destination_lease.release().await?;
    Ok(())
}
