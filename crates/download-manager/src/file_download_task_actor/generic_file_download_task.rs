use std::{
    fmt,
    marker::PhantomData,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use tokio::{
    sync::{
        Mutex as TokioMutex,
        broadcast::{Sender as TokioBroadcastSender, channel as tokio_broadcast_channel},
        mpsc::{Sender as TokioMpscSender, channel as tokio_mpsc_channel},
        oneshot::channel as tokio_oneshot_channel,
        watch::{Receiver as TokioWatchReceiver, channel as tokio_watch_channel},
    },
    task::JoinHandle as TokioJoinHandle,
};
use tokio_stream::{StreamExt as TokioStreamExt, wrappers::BroadcastStream as TokioBroadcastStream};

use crate::{
    DownloadError, DownloadEventSender, DownloadId, FileCheck, FileDownloadEvent, FileDownloadPhase, FileDownloadState,
    backends::common::{Backend, InitialTaskAttachment},
    file_download_task::{FileDownloadTask, ManagedFileDownloadTask},
    file_download_task_actor::{
        DownloadActorState, DownloadTaskActor, PendingProgressSlot, ProgressCounters, PublicProjection, TaskCommand,
        TerminalOutcome, project_runtime_public_state,
    },
    lock_manager::DestinationLockLease,
    reducer::InitialLifecycleState,
    traits::{ActiveDownloadGenerationCounter, BackendEventSender, DownloadBackend, DownloadConfig},
};

pub struct GenericFileDownloadTask<B: DownloadBackend> {
    config: Arc<DownloadConfig>,
    command_sender: TokioMpscSender<TaskCommand>,
    public_state_receiver: TokioWatchReceiver<FileDownloadState>,
    progress_sender: TokioBroadcastSender<FileDownloadState>,
    terminal_receiver: TokioWatchReceiver<TerminalOutcome>,
    listener_task: Arc<TokioMutex<Option<TokioJoinHandle<()>>>>,
    is_stopped: AtomicBool,
    backend: PhantomData<B>,
}

impl<B: DownloadBackend> GenericFileDownloadTask<B> {
    pub(crate) async fn spawn_with_initial_attachment(
        config: Arc<DownloadConfig>,
        context: Arc<B::Context>,
        initial_lifecycle_state: InitialLifecycleState,
        initial_projection: PublicProjection,
        initial_progress: ProgressCounters,
        mut startup_lease: Option<DestinationLockLease>,
    ) -> Result<Self, DownloadError>
    where
        B: Backend,
    {
        let (command_sender, command_receiver) = tokio_mpsc_channel(64);
        let (backend_event_sender, backend_event_receiver) = tokio_mpsc_channel(64);
        let pending_progress = Arc::new(TokioMutex::new(PendingProgressSlot::default()));
        let (progress_waker_sender, progress_waker_receiver) = tokio_watch_channel(());
        let backend_event_sender =
            BackendEventSender::new(backend_event_sender, Arc::clone(&pending_progress), progress_waker_sender);
        let (progress_sender, _) = tokio_broadcast_channel(64);
        let (terminal_sender, terminal_receiver) = tokio_watch_channel(TerminalOutcome::Pending);

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
                InitialLifecycleState::Downloaded {
                    ..
                },
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
            Arc::clone(&context),
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
            terminal_sender,
        );
        tokio::spawn(actor.run());

        Ok(Self {
            config,
            command_sender,
            public_state_receiver,
            progress_sender,
            terminal_receiver,
            listener_task: Arc::new(TokioMutex::new(None)),
            is_stopped: AtomicBool::new(false),
            backend: PhantomData,
        })
    }

    async fn send_command(
        &self,
        command_builder: impl FnOnce(tokio::sync::oneshot::Sender<Result<(), DownloadError>>) -> TaskCommand,
    ) -> Result<(), DownloadError> {
        if self.is_stopped.load(Ordering::SeqCst) {
            return Err(DownloadError::TaskStopped);
        }
        let (reply_sender, reply_receiver) = tokio_oneshot_channel();
        self.command_sender.send(command_builder(reply_sender)).await.map_err(|_| DownloadError::ChannelClosed)?;
        reply_receiver.await.unwrap_or(Err(DownloadError::TaskStopped))
    }

    async fn wait_for_actor_stopped(&self) {
        let mut terminal_receiver = self.terminal_receiver.clone();
        loop {
            if matches!(terminal_receiver.borrow().clone(), TerminalOutcome::ActorStopped) {
                break;
            }

            if terminal_receiver.changed().await.is_err() {
                break;
            }
        }
    }
}

impl<B: DownloadBackend> fmt::Debug for GenericFileDownloadTask<B> {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        formatter
            .debug_struct("GenericFileDownloadTask")
            .field("download_id", &self.config.download_id)
            .field("source_url", &self.config.source_url)
            .field("destination", &self.config.destination)
            .finish()
    }
}

#[async_trait::async_trait]
impl<B: DownloadBackend> crate::FileDownloadTask for GenericFileDownloadTask<B> {
    fn download_id(&self) -> DownloadId {
        self.config.download_id
    }

    fn source_url(&self) -> &str {
        &self.config.source_url
    }

    fn destination(&self) -> &Path {
        &self.config.destination
    }

    fn file_check(&self) -> &FileCheck {
        &self.config.file_check
    }

    fn expected_bytes(&self) -> Option<u64> {
        self.config.expected_bytes
    }

    async fn download(&self) -> Result<(), DownloadError> {
        self.send_command(|reply_sender| TaskCommand::Download {
            reply_sender,
        })
        .await
    }

    async fn pause(&self) -> Result<(), DownloadError> {
        self.send_command(|reply_sender| TaskCommand::Pause {
            reply_sender,
        })
        .await
    }

    async fn cancel(&self) -> Result<(), DownloadError> {
        self.send_command(|reply_sender| TaskCommand::Cancel {
            reply_sender,
        })
        .await
    }

    async fn state(&self) -> FileDownloadState {
        self.public_state_receiver.borrow().clone()
    }

    async fn progress(&self) -> Result<TokioBroadcastStream<FileDownloadState>, DownloadError> {
        Ok(TokioBroadcastStream::new(self.progress_sender.subscribe()))
    }

    async fn start_listening(
        &self,
        global_broadcast: DownloadEventSender,
    ) {
        let mut listener_task = self.listener_task.lock().await;
        if listener_task.is_some() {
            return;
        }

        let download_id = self.config.download_id;
        let destination = self.config.destination.clone();
        let mut local_stream = TokioBroadcastStream::new(self.progress_sender.subscribe());
        *listener_task = Some(tokio::spawn(async move {
            let mut last_downloaded_bytes = 0u64;

            while let Some(result) = local_stream.next().await {
                let Ok(state) = result else {
                    continue;
                };

                match state.phase {
                    FileDownloadPhase::Downloading => {
                        let bytes_written = state.downloaded_bytes.saturating_sub(last_downloaded_bytes);
                        last_downloaded_bytes = state.downloaded_bytes;
                        let event = FileDownloadEvent::ProgressUpdate {
                            bytes_written,
                            total_bytes_written: state.downloaded_bytes,
                            total_bytes_expected: state.total_bytes,
                        };
                        let _ = global_broadcast.send((download_id, event));
                    },
                    FileDownloadPhase::Downloaded => {
                        let event = FileDownloadEvent::DownloadCompleted {
                            tmp_path: destination.clone(),
                            final_destination: destination.clone(),
                        };
                        let _ = global_broadcast.send((download_id, event));
                        break;
                    },
                    FileDownloadPhase::Error(message) => {
                        let event = FileDownloadEvent::Error {
                            message,
                        };
                        let _ = global_broadcast.send((download_id, event));
                        break;
                    },
                    FileDownloadPhase::NotDownloaded
                    | FileDownloadPhase::Paused
                    | FileDownloadPhase::LockedByOther(_) => {},
                }
            }
        }));
    }

    async fn stop_listening(&self) {
        if let Some(listener_task) = self.listener_task.lock().await.take() {
            listener_task.abort();
            let _ = listener_task.await;
        }
    }

    async fn wait(&self) {
        let mut terminal_receiver = self.terminal_receiver.clone();
        loop {
            match terminal_receiver.borrow().clone() {
                TerminalOutcome::Pending => {},
                TerminalOutcome::Downloaded | TerminalOutcome::Error(_) | TerminalOutcome::ActorStopped => break,
            }

            if terminal_receiver.changed().await.is_err() {
                break;
            }
        }
    }

    fn broadcast_sender(&self) -> TokioBroadcastSender<FileDownloadState> {
        self.progress_sender.clone()
    }
}

#[async_trait::async_trait]
impl<B: DownloadBackend> ManagedFileDownloadTask for GenericFileDownloadTask<B> {
    async fn shutdown_for_removal(&self) -> Result<(), DownloadError> {
        if self.is_stopped.swap(true, Ordering::SeqCst) {
            self.wait_for_actor_stopped().await;
            self.stop_listening().await;
            return Ok(());
        }

        let (reply_sender, reply_receiver) = tokio_oneshot_channel();
        if self
            .command_sender
            .send(TaskCommand::Remove {
                reply_sender,
            })
            .await
            .is_err()
        {
            self.wait_for_actor_stopped().await;
            self.stop_listening().await;
            return Ok(());
        }

        let result = reply_receiver.await.unwrap_or(Err(DownloadError::TaskStopped));
        self.wait_for_actor_stopped().await;
        self.stop_listening().await;
        result
    }

    fn is_stopped(&self) -> bool {
        self.is_stopped.load(Ordering::SeqCst)
    }
}

impl<B: DownloadBackend> Drop for GenericFileDownloadTask<B> {
    fn drop(&mut self) {}
}
