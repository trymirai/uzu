use std::{path::PathBuf, sync::Arc};

use tokio::{
    runtime::Handle as TokioHandle,
    sync::{Mutex as TokioMutex, mpsc::channel as tokio_mpsc_channel, watch::channel as tokio_watch_channel},
};
use uuid::Uuid;

use crate::{
    FileCheck,
    backends::universal::{UniversalActiveTask, UniversalBackend, UniversalBackendContext},
    file_download_task_actor::{DispatchContext, DownloadFsm, DownloadLifecycleState, FsmEvent, PendingProgressSlot},
    traits::{ActiveDownloadGeneration, BackendEventSender, DownloadConfig},
};

fn backend_event_sender() -> BackendEventSender {
    let (backend_event_sender, _backend_event_receiver) = tokio_mpsc_channel(64);
    let pending_progress = Arc::new(TokioMutex::new(PendingProgressSlot::default()));
    let (progress_waker_sender, _progress_waker_receiver) = tokio_watch_channel(());
    BackendEventSender::new(backend_event_sender, pending_progress, progress_waker_sender)
}

#[tokio::test]
async fn downloading_ignores_stale_completed_generation() {
    let current_generation = ActiveDownloadGeneration::new(1);
    let stale_generation = ActiveDownloadGeneration::new(0);
    let config = Arc::new(DownloadConfig {
        download_id: Uuid::new_v4(),
        source_url: "http://example.test/model.bin".to_string(),
        destination: PathBuf::from("model.bin"),
        file_check: FileCheck::None,
        expected_bytes: Some(100),
        manager_id: "test-manager".to_string(),
        manager_instance_id: Uuid::new_v4(),
    });
    let active_task = UniversalActiveTask::new(Vec::new().into_boxed_slice(), PathBuf::from("model.bin.part"));
    let download_fsm = DownloadFsm::<UniversalBackend>::new(
        Arc::clone(&config),
        Arc::new(UniversalBackendContext::new(TokioHandle::current())),
        backend_event_sender(),
    );
    let mut state_machine = download_fsm.into_state_machine(DownloadLifecycleState::Downloading {
        active_task: Some(active_task),
        generation: current_generation,
    });
    let mut dispatch_context = DispatchContext::new(None);

    state_machine
        .handle_with_context(
            &FsmEvent::BackendCompleted {
                generation: stale_generation,
            },
            &mut dispatch_context,
        )
        .await;

    assert!(dispatch_context.effects.is_empty());
    assert!(matches!(state_machine.state(), DownloadLifecycleState::Downloading { .. }));
}
