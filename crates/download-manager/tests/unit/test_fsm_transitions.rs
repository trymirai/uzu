use std::{path::PathBuf, sync::Arc};

use download_manager::{
    DownloadError, FileCheck,
    backends::universal::{UniversalActiveTask, UniversalBackend, UniversalBackendContext},
    file_download_task_actor::{
        DispatchContext, DownloadActorEffect, DownloadFsm, DownloadLifecycleState, FsmEvent, PendingProgressSlot,
    },
    traits::{ActiveDownloadGeneration, BackendEventSender, DownloadConfig},
};
use tokio::sync::{
    Mutex as TokioMutex,
    mpsc::channel as tokio_mpsc_channel,
    watch::channel as tokio_watch_channel,
};
use uuid::Uuid;

use crate::common::MockRegistry;

async fn fsm(
    initial_state: DownloadLifecycleState<UniversalBackend>
) -> Result<
    (
        statig::awaitable::StateMachine<DownloadFsm<UniversalBackend>>,
        Arc<DownloadConfig>,
    ),
    Box<dyn std::error::Error>,
> {
    let registry = MockRegistry::start().await?;
    let served_file = registry.file("config.json")?;
    let config = Arc::new(DownloadConfig {
        download_id: Uuid::nil(),
        source_url: served_file.file.url.clone(),
        destination: PathBuf::from(served_file.file.name.clone()),
        file_check: FileCheck::None,
        expected_bytes: Some(served_file.file.size as u64),
        manager_id: "test-manager".to_string(),
    });
    let (backend_event_sender, _backend_event_receiver) = tokio_mpsc_channel(64);
    let pending_progress = Arc::new(TokioMutex::new(PendingProgressSlot::default()));
    let (progress_waker_sender, _progress_waker_receiver) = tokio_watch_channel(());
    let backend_event_sender =
        BackendEventSender::new(backend_event_sender, pending_progress, progress_waker_sender);
    let download_fsm =
        DownloadFsm::<UniversalBackend>::new(Arc::clone(&config), Arc::new(UniversalBackendContext::default()), backend_event_sender);
    Ok((download_fsm.into_state_machine(initial_state), config))
}

#[tokio::test]
async fn test_fsm_idle_pause_is_invalid_from_not_downloaded() -> Result<(), Box<dyn std::error::Error>> {
    let (mut fsm, _) = fsm(DownloadLifecycleState::NotDownloaded {}).await?;
    let mut dispatch_context = DispatchContext::new(None);

    fsm.handle_with_context(&FsmEvent::Pause, &mut dispatch_context).await;

    let reply = reply_effect(dispatch_context.effects).expect("pause must reply");
    assert_eq!(reply, Err(DownloadError::InvalidStateTransition));
    assert!(matches!(fsm.state(), DownloadLifecycleState::NotDownloaded {}));
    Ok(())
}

#[tokio::test]
async fn test_fsm_idle_cancel_is_ok_from_downloaded() -> Result<(), Box<dyn std::error::Error>> {
    let (mut fsm, config) = fsm(DownloadLifecycleState::Downloaded {
        file_path: PathBuf::from("file.bin"),
        crc_path: None,
    })
    .await?;
    let mut dispatch_context = DispatchContext::new(None);

    fsm.handle_with_context(&FsmEvent::Cancel, &mut dispatch_context).await;

    let reply = reply_effect(dispatch_context.effects).expect("cancel must reply");
    assert_eq!(reply, Ok(()));
    assert!(matches!(fsm.state(), DownloadLifecycleState::Downloaded { .. }));
    assert_eq!(config.expected_bytes, Some(55559));
    Ok(())
}

#[tokio::test]
async fn test_fsm_downloading_ignores_stale_completed_generation() -> Result<(), Box<dyn std::error::Error>> {
    let active_task = UniversalActiveTask::new(Box::from([]), PathBuf::from("model.bin.part"));
    let current_generation = ActiveDownloadGeneration::new(1);
    let stale_generation = ActiveDownloadGeneration::new(0);
    let (mut fsm, _) =
        fsm(DownloadLifecycleState::Downloading {
            active_task: Some(active_task),
            generation: current_generation,
        })
        .await?;
    let mut dispatch_context = DispatchContext::new(None);

    fsm.handle_with_context(
        &FsmEvent::BackendCompleted {
            generation: stale_generation,
        },
        &mut dispatch_context,
    )
    .await;

    assert!(dispatch_context.effects.is_empty());
    assert!(matches!(fsm.state(), DownloadLifecycleState::Downloading { .. }));
    Ok(())
}

fn reply_effect(
    effects: Vec<DownloadActorEffect<UniversalBackend>>
) -> Option<Result<(), DownloadError>> {
    effects.into_iter().find_map(|effect| match effect {
        DownloadActorEffect::Reply(result) => Some(result),
        _ => None,
    })
}
