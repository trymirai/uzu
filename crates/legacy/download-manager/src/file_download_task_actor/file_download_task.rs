use std::sync::Arc;

use kiban::rt;
use tokio::sync::{
    Mutex as TokioMutex,
    broadcast::{Sender as TokioBroadcastSender, channel as tokio_broadcast_channel},
    mpsc::{Sender as TokioMpscSender, channel as tokio_mpsc_channel},
    oneshot::{Sender as TokioOneshotSender, channel as tokio_oneshot_channel},
    watch::{Receiver as TokioWatchReceiver, channel as tokio_watch_channel},
};
use tokio_stream::wrappers::BroadcastStream;
use uuid::Uuid;

use crate::{
    DownloadError, DownloadState, DownloadTaskRequest, LockFileState,
    backends::common::{
        ActiveDownloadGenerationCounter, Backend, BackendEventSender, DownloadConfig, InitialTaskAttachment,
    },
    check_lock_file,
    file_download_task_actor::{
        DownloadActorState, DownloadTaskActor, PendingProgressSlot, ProgressCounters, TaskCommand,
        project_runtime_public_state,
    },
    lock_manager::{DestinationLockLease, lock_path_for_destination},
    reducer::{Decision, InitialLifecycleState},
};

pub struct FileDownloadTask {
    pub request: DownloadTaskRequest,
    manager_id: String,
    manager_instance_id: Uuid,
    command_sender: TokioMpscSender<TaskCommand>,
    public_state_receiver: TokioWatchReceiver<DownloadState>,
    progress_sender: TokioBroadcastSender<DownloadState>,
}

impl FileDownloadTask {
    pub fn state(&self) -> DownloadState {
        self.public_state_receiver.borrow().clone()
    }

    pub fn progress(&self) -> BroadcastStream<DownloadState> {
        BroadcastStream::new(self.progress_sender.subscribe())
    }

    pub async fn download(&self) -> Result<(), DownloadError> {
        self.send_command(|reply_sender| TaskCommand::Download {
            reply_sender,
        })
        .await
    }

    pub async fn pause(&self) -> Result<(), DownloadError> {
        self.send_command(|reply_sender| TaskCommand::Pause {
            reply_sender,
        })
        .await
    }

    pub async fn delete(&self) -> Result<(), DownloadError> {
        self.send_command(|reply_sender| TaskCommand::Delete {
            reply_sender,
        })
        .await
    }

    pub async fn foreign_lock(&self) -> Option<String> {
        let lock_path = lock_path_for_destination(&self.request.destination);
        match check_lock_file(&lock_path, &self.manager_id, self.manager_instance_id, kiban::process::id()).await {
            LockFileState::OwnedByOtherApp(lock_file_info) => Some(lock_file_info.manager_id),
            _ => None,
        }
    }

    async fn send_command(
        &self,
        command: impl FnOnce(TokioOneshotSender<Result<(), DownloadError>>) -> TaskCommand,
    ) -> Result<(), DownloadError> {
        let (reply_sender, reply_receiver) = tokio_oneshot_channel();
        self.command_sender.send(command(reply_sender)).await.map_err(|_| DownloadError::ChannelClosed)?;
        reply_receiver.await.unwrap_or(Err(DownloadError::ChannelClosed))
    }
}

pub async fn spawn_file_download_task<B: Backend>(
    request: DownloadTaskRequest,
    config: Arc<DownloadConfig>,
    context: Arc<B::Context>,
    decision: Decision,
    mut startup_lease: Option<DestinationLockLease>,
) -> Result<(FileDownloadTask, TokioWatchReceiver<DownloadState>), DownloadError> {
    let Decision {
        initial_lifecycle_state,
        initial_projection,
        initial_progress,
        ..
    } = decision;
    let (command_sender, command_receiver) = tokio_mpsc_channel(64);
    let (backend_event_sender, backend_event_receiver) = tokio_mpsc_channel(64);
    let pending_progress = Arc::new(TokioMutex::new(PendingProgressSlot::default()));
    let (progress_waker_sender, progress_waker_receiver) = tokio_watch_channel(());
    let backend_event_sender = BackendEventSender::new(
        config.download_id,
        backend_event_sender,
        Arc::clone(&pending_progress),
        progress_waker_sender,
    );
    let (progress_sender, _) = tokio_broadcast_channel(64);

    let mut generation_counter = ActiveDownloadGenerationCounter::default();
    let attachment_generation = generation_counter.allocate_next();
    let attachment = if let Some(destination_lease) = startup_lease.as_ref() {
        match B::initial_task_attachment(
            context.as_ref(),
            Arc::clone(&config),
            attachment_generation,
            backend_event_sender.clone(),
            destination_lease,
        )
        .await
        {
            Ok(attachment) => attachment,
            Err(error) => {
                if let Some(lease) = startup_lease.take() {
                    let _ = lease.release().await;
                }
                return Err(error);
            },
        }
    } else {
        InitialTaskAttachment::None
    };
    let (lifecycle_state, progress_counters) = match (attachment, &initial_lifecycle_state) {
        (InitialTaskAttachment::None, _) => (initial_lifecycle_state.into(), initial_progress),
        (
            InitialTaskAttachment::Downloading {
                ..
            },
            InitialLifecycleState::Downloaded,
        ) => (initial_lifecycle_state.into(), initial_progress),
        (
            InitialTaskAttachment::Downloading {
                active_task,
                initial_downloaded_bytes,
                total_bytes,
            },
            _,
        ) => {
            let Some(destination_lease) = startup_lease.take() else {
                return Err(DownloadError::Backend(
                    "backend returned an attached task without a startup lease".to_string(),
                ));
            };
            (
                DownloadActorState::Downloading {
                    active_task,
                    generation: attachment_generation,
                    destination_lease,
                },
                ProgressCounters {
                    downloaded_bytes: initial_downloaded_bytes,
                    total_bytes: total_bytes.or(config.expected_bytes).unwrap_or(initial_downloaded_bytes),
                },
            )
        },
    };

    if let Some(lease) = startup_lease.take() {
        lease.release().await?;
    }

    let initial_public_state =
        project_runtime_public_state(&lifecycle_state, &initial_projection, progress_counters, &config);
    let (public_state_sender, public_state_receiver) = tokio_watch_channel(initial_public_state);

    let actor = DownloadTaskActor::<B>::new(
        Arc::clone(&config),
        context,
        backend_event_sender,
        generation_counter,
        lifecycle_state,
        initial_projection,
        progress_counters,
        command_receiver,
        backend_event_receiver,
        pending_progress,
        progress_waker_receiver,
        public_state_sender,
        progress_sender.clone(),
    );
    rt::spawn(actor.run());

    let actor = public_state_receiver.clone();
    Ok((
        FileDownloadTask {
            request,
            manager_id: config.manager_id.clone(),
            manager_instance_id: config.manager_instance_id,
            command_sender,
            public_state_receiver,
            progress_sender,
        },
        actor,
    ))
}
