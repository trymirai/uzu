use std::{fmt, marker::PhantomData, path::Path, sync::Arc};

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
    file_download_task_actor::{
        DownloadTaskActor, LifecycleState, PendingProgressSlot, ProgressCounters, PublicProjection, TaskCommand,
        TerminalOutcome, project_public_state,
    },
    reducer::InitialLifecycleState,
    traits::{ActiveDownloadGenerationCounter, BackendEventSender, DownloadBackend, DownloadConfig},
};

pub struct GenericFileDownloadTask<B: DownloadBackend> {
    config: Arc<DownloadConfig>,
    command_sender: TokioMpscSender<TaskCommand>,
    actor_task: TokioJoinHandle<()>,
    public_state_receiver: TokioWatchReceiver<FileDownloadState>,
    progress_sender: TokioBroadcastSender<FileDownloadState>,
    terminal_receiver: TokioWatchReceiver<TerminalOutcome>,
    listener_task: Arc<TokioMutex<Option<TokioJoinHandle<()>>>>,
    backend: PhantomData<B>,
}

impl<B: DownloadBackend> GenericFileDownloadTask<B> {
    pub fn spawn(
        config: Arc<DownloadConfig>,
        context: Arc<B::Context>,
        initial_lifecycle_state: InitialLifecycleState,
        initial_projection: PublicProjection,
        initial_progress: ProgressCounters,
    ) -> Self {
        let initial_public_state =
            project_public_state(&initial_lifecycle_state, &initial_projection, initial_progress, &config);
        let (command_sender, command_receiver) = tokio_mpsc_channel(64);
        let (backend_event_sender, backend_event_receiver) = tokio_mpsc_channel(64);
        let pending_progress = Arc::new(TokioMutex::new(PendingProgressSlot::default()));
        let (progress_waker_sender, progress_waker_receiver) = tokio_watch_channel(());
        let backend_event_sender =
            BackendEventSender::new(backend_event_sender, Arc::clone(&pending_progress), progress_waker_sender);
        let (public_state_sender, public_state_receiver) = tokio_watch_channel(initial_public_state.clone());
        let (progress_sender, _) = tokio_broadcast_channel(64);
        let (terminal_sender, terminal_receiver) = tokio_watch_channel(TerminalOutcome::Pending);

        let actor = DownloadTaskActor::<B>::new(
            Arc::clone(&config),
            initial_lifecycle_state.into(),
            initial_projection,
            initial_progress,
            ActiveDownloadGenerationCounter::default(),
            context,
            command_receiver,
            backend_event_receiver,
            pending_progress,
            progress_waker_receiver,
            backend_event_sender,
            public_state_sender,
            progress_sender.clone(),
            terminal_sender,
        );
        let actor_task = tokio::spawn(actor.run());

        Self {
            config,
            command_sender,
            actor_task,
            public_state_receiver,
            progress_sender,
            terminal_receiver,
            listener_task: Arc::new(TokioMutex::new(None)),
            backend: PhantomData,
        }
    }

    async fn send_command(
        &self,
        command_builder: impl FnOnce(tokio::sync::oneshot::Sender<Result<(), DownloadError>>) -> TaskCommand,
    ) -> Result<(), DownloadError> {
        let (reply_sender, reply_receiver) = tokio_oneshot_channel();
        self.command_sender.send(command_builder(reply_sender)).await.map_err(|_| DownloadError::ChannelClosed)?;
        reply_receiver.await.unwrap_or(Err(DownloadError::TaskStopped))
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
                        let _ = global_broadcast.send((
                            download_id,
                            FileDownloadEvent::ProgressUpdate {
                                bytes_written,
                                total_bytes_written: state.downloaded_bytes,
                                total_bytes_expected: state.total_bytes,
                            },
                        ));
                    },
                    FileDownloadPhase::Downloaded => {
                        let _ = global_broadcast.send((
                            download_id,
                            FileDownloadEvent::DownloadCompleted {
                                tmp_path: destination.clone(),
                                final_destination: destination.clone(),
                            },
                        ));
                        break;
                    },
                    FileDownloadPhase::Error(message) => {
                        let _ = global_broadcast.send((
                            download_id,
                            FileDownloadEvent::Error {
                                message,
                            },
                        ));
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

impl<B: DownloadBackend> Drop for GenericFileDownloadTask<B> {
    fn drop(&mut self) {
        self.actor_task.abort();
    }
}

impl<B: DownloadBackend> From<InitialLifecycleState> for LifecycleState<B> {
    fn from(initial_lifecycle_state: InitialLifecycleState) -> Self {
        match initial_lifecycle_state {
            InitialLifecycleState::NotDownloaded => Self::NotDownloaded,
            InitialLifecycleState::Paused {
                part_path,
            } => Self::Paused {
                part_path,
            },
            InitialLifecycleState::Downloaded {
                file_path,
                crc_path,
            } => Self::Downloaded {
                file_path,
                crc_path,
            },
        }
    }
}
