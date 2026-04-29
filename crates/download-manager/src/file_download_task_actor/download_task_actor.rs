use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use tokio::sync::{
    Mutex as TokioMutex,
    broadcast::Sender as TokioBroadcastSender,
    mpsc::Receiver as TokioMpscReceiver,
    watch::{Receiver as TokioWatchReceiver, Sender as TokioWatchSender},
};

use crate::{
    DownloadError, FileCheck, FileDownloadState, LockFileState, acquire_lock, check_lock_file,
    crc_utils::{calculate_and_verify_crc, crc_path_for_file, save_crc_file},
    file_download_task_actor::{
        BackendEvent, LifecycleState, PendingProgressSlot, ProgressCounters, PublicProjection, TaskCommand,
        TerminalOutcome, project_runtime_public_state,
    },
    release_lock,
    traits::{
        ActiveDownloadGenerationCounter, ActiveTask, BackendContext, BackendEventSender, DownloadBackend,
        DownloadConfig,
    },
};

pub struct DownloadTaskActor<B: DownloadBackend> {
    config: Arc<DownloadConfig>,
    lifecycle_state: LifecycleState<B>,
    projection: PublicProjection,
    progress_counters: ProgressCounters,
    generation_counter: ActiveDownloadGenerationCounter,
    context: Arc<B::Context>,
    command_receiver: TokioMpscReceiver<TaskCommand>,
    backend_event_receiver: TokioMpscReceiver<BackendEvent>,
    pending_progress: Arc<TokioMutex<PendingProgressSlot>>,
    progress_waker_receiver: TokioWatchReceiver<()>,
    backend_event_sender: BackendEventSender,
    public_state_sender: TokioWatchSender<FileDownloadState>,
    progress_sender: TokioBroadcastSender<FileDownloadState>,
    terminal_sender: TokioWatchSender<TerminalOutcome>,
}

impl<B: DownloadBackend> DownloadTaskActor<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: Arc<DownloadConfig>,
        lifecycle_state: LifecycleState<B>,
        projection: PublicProjection,
        progress_counters: ProgressCounters,
        generation_counter: ActiveDownloadGenerationCounter,
        context: Arc<B::Context>,
        command_receiver: TokioMpscReceiver<TaskCommand>,
        backend_event_receiver: TokioMpscReceiver<BackendEvent>,
        pending_progress: Arc<TokioMutex<PendingProgressSlot>>,
        progress_waker_receiver: TokioWatchReceiver<()>,
        backend_event_sender: BackendEventSender,
        public_state_sender: TokioWatchSender<FileDownloadState>,
        progress_sender: TokioBroadcastSender<FileDownloadState>,
        terminal_sender: TokioWatchSender<TerminalOutcome>,
    ) -> Self {
        Self {
            config,
            lifecycle_state,
            projection,
            progress_counters,
            generation_counter,
            context,
            command_receiver,
            backend_event_receiver,
            pending_progress,
            progress_waker_receiver,
            backend_event_sender,
            public_state_sender,
            progress_sender,
            terminal_sender,
        }
    }

    pub async fn run(mut self) {
        self.publish_current_state();

        loop {
            tokio::select! {
                command = self.command_receiver.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    self.handle_command(command).await;
                }
                backend_event = self.backend_event_receiver.recv() => {
                    let Some(backend_event) = backend_event else {
                        break;
                    };
                    self.handle_backend_event(backend_event).await;
                }
                progress_wake_result = self.progress_waker_receiver.changed() => {
                    if progress_wake_result.is_err() {
                        break;
                    }
                    self.handle_pending_progress().await;
                }
            }

            self.publish_current_state();
        }

        let _ = self.terminal_sender.send(TerminalOutcome::ActorStopped);
    }

    async fn handle_command(
        &mut self,
        command: TaskCommand,
    ) {
        match command {
            TaskCommand::Download {
                reply_sender,
            } => {
                let _ = reply_sender.send(self.handle_download().await);
            },
            TaskCommand::Pause {
                reply_sender,
            } => {
                let _ = reply_sender.send(self.handle_pause().await);
            },
            TaskCommand::Cancel {
                reply_sender,
            } => {
                let _ = reply_sender.send(self.handle_cancel().await);
            },
        }
    }

    async fn handle_download(&mut self) -> Result<(), DownloadError> {
        match &self.lifecycle_state {
            LifecycleState::Downloaded {
                ..
            }
            | LifecycleState::Downloading {
                ..
            } => Ok(()),
            LifecycleState::NotDownloaded => {
                self.acquire_destination_lock().await?;
                let generation = self.generation_counter.allocate_next();
                self.progress_counters = ProgressCounters {
                    downloaded_bytes: 0,
                    total_bytes: self.config.expected_bytes.unwrap_or(0),
                };
                let active_task = self
                    .context
                    .download(Arc::clone(&self.config), generation, self.backend_event_sender.clone())
                    .await
                    .map_err(|error| DownloadError::Backend(error.to_string()))?;
                self.projection = PublicProjection::None;
                self.lifecycle_state = LifecycleState::Downloading {
                    active_task: Some(active_task),
                    generation,
                };
                Ok(())
            },
            LifecycleState::Paused {
                part_path,
            } => {
                let part_path = part_path.clone();
                self.acquire_destination_lock().await?;
                let generation = self.generation_counter.allocate_next();
                let resume_bytes = file_size(&part_path).unwrap_or(0);
                self.progress_counters = ProgressCounters {
                    downloaded_bytes: resume_bytes,
                    total_bytes: self.config.expected_bytes.unwrap_or(resume_bytes),
                };
                let active_task = self
                    .context
                    .resume(Arc::clone(&self.config), generation, &part_path, self.backend_event_sender.clone())
                    .await
                    .map_err(|error| DownloadError::Backend(error.to_string()))?;
                self.projection = PublicProjection::None;
                self.lifecycle_state = LifecycleState::Downloading {
                    active_task: Some(active_task),
                    generation,
                };
                Ok(())
            },
        }
    }

    async fn handle_pause(&mut self) -> Result<(), DownloadError> {
        match &mut self.lifecycle_state {
            LifecycleState::Paused {
                ..
            } => Ok(()),
            LifecycleState::NotDownloaded
            | LifecycleState::Downloaded {
                ..
            } => Err(DownloadError::InvalidStateTransition),
            LifecycleState::Downloading {
                active_task,
                ..
            } => match active_task.take() {
                Some(active_task) => match active_task.pause(&self.config.destination).await {
                    Ok(part_path) => {
                        let downloaded_bytes = file_size(&part_path).unwrap_or(self.progress_counters.downloaded_bytes);
                        self.progress_counters = ProgressCounters {
                            downloaded_bytes,
                            total_bytes: self.config.expected_bytes.unwrap_or(downloaded_bytes),
                        };
                        self.lifecycle_state = LifecycleState::Paused {
                            part_path,
                        };
                        self.release_destination_lock().await;
                        Ok(())
                    },
                    Err(error) => {
                        let message = error.to_string();
                        self.projection = PublicProjection::StickyError(message.clone());
                        self.lifecycle_state = LifecycleState::NotDownloaded;
                        self.progress_counters = ProgressCounters::default();
                        self.release_destination_lock().await;
                        let _ = self.terminal_sender.send(TerminalOutcome::Error(message.clone()));
                        Err(DownloadError::Backend(message))
                    },
                },
                None => {
                    let message = "active task missing while downloading".to_string();
                    self.projection = PublicProjection::StickyError(message.clone());
                    self.lifecycle_state = LifecycleState::NotDownloaded;
                    self.progress_counters = ProgressCounters::default();
                    self.release_destination_lock().await;
                    let _ = self.terminal_sender.send(TerminalOutcome::Error(message.clone()));
                    Err(DownloadError::Backend(message))
                },
            },
        }
    }

    async fn handle_cancel(&mut self) -> Result<(), DownloadError> {
        match &mut self.lifecycle_state {
            LifecycleState::Downloading {
                active_task,
                ..
            } => {
                if let Some(active_task) = active_task.take() {
                    let _ = active_task.cancel(&self.config.destination).await;
                }
                remove_resume_artifact(&self.config.destination);
                self.lifecycle_state = LifecycleState::NotDownloaded;
                self.release_destination_lock().await;
            },
            LifecycleState::Paused {
                part_path,
            } => {
                remove_file(part_path);
                self.lifecycle_state = LifecycleState::NotDownloaded;
            },
            LifecycleState::Downloaded {
                ..
            } => {
                self.lifecycle_state = LifecycleState::NotDownloaded;
            },
            LifecycleState::NotDownloaded => {},
        }

        self.progress_counters = ProgressCounters::default();
        self.projection = PublicProjection::None;
        Ok(())
    }

    async fn handle_backend_event(
        &mut self,
        backend_event: BackendEvent,
    ) {
        if !self.lifecycle_state.is_downloading_generation(backend_event.generation()) {
            return;
        }

        match backend_event {
            BackendEvent::Completed {
                ..
            } => self.handle_completion().await,
            BackendEvent::Error {
                message,
                ..
            } => {
                self.projection = PublicProjection::StickyError(message.clone());
                self.lifecycle_state = LifecycleState::NotDownloaded;
                self.progress_counters = ProgressCounters::default();
                self.release_destination_lock().await;
                let _ = self.terminal_sender.send(TerminalOutcome::Error(message));
            },
        }
    }

    async fn handle_pending_progress(&mut self) {
        let progress = self.pending_progress.lock().await.take();
        let Some(progress) = progress else {
            return;
        };

        if self.lifecycle_state.is_downloading_generation(progress.generation) {
            self.progress_counters = ProgressCounters {
                downloaded_bytes: progress.downloaded_bytes,
                total_bytes: progress
                    .total_bytes
                    .unwrap_or_else(|| self.config.expected_bytes.unwrap_or(progress.downloaded_bytes)),
            };
        }
    }

    async fn handle_completion(&mut self) {
        match validate_completed_file(&self.config).await {
            Ok(total_bytes) => {
                self.lifecycle_state = LifecycleState::Downloaded {
                    file_path: self.config.destination.clone(),
                    crc_path: crc_path_for_destination(&self.config.destination),
                };
                self.progress_counters = ProgressCounters {
                    downloaded_bytes: total_bytes,
                    total_bytes,
                };
                self.projection = PublicProjection::None;
                self.release_destination_lock().await;
                let _ = self.terminal_sender.send(TerminalOutcome::Downloaded);
            },
            Err(message) => {
                remove_file(&self.config.destination);
                remove_resume_artifact(&self.config.destination);
                self.lifecycle_state = LifecycleState::NotDownloaded;
                self.progress_counters = ProgressCounters::default();
                self.projection = PublicProjection::StickyError(message.clone());
                self.release_destination_lock().await;
                let _ = self.terminal_sender.send(TerminalOutcome::Error(message));
            },
        }
    }

    fn publish_current_state(&self) {
        let public_state =
            project_runtime_public_state(&self.lifecycle_state, &self.projection, self.progress_counters, &self.config);
        let _ = self.public_state_sender.send(public_state.clone());
        let _ = self.progress_sender.send(public_state);
    }

    async fn acquire_destination_lock(&mut self) -> Result<(), DownloadError> {
        let lock_path = lock_path_for_destination(&self.config.destination);
        match check_lock_file(&lock_path, &self.config.manager_id, std::process::id()) {
            LockFileState::OwnedByOtherApp(lock_file_info) => {
                self.projection = PublicProjection::LockedByOther(lock_file_info.manager_id.clone());
                Err(DownloadError::LockedByOther(lock_file_info.manager_id))
            },
            LockFileState::Missing
            | LockFileState::OwnedByUs(_)
            | LockFileState::OwnedBySameAppOldProcess(_)
            | LockFileState::Stale(_) => {
                acquire_lock(&lock_path, &self.config.manager_id).await?;
                Ok(())
            },
        }
    }

    async fn release_destination_lock(&self) {
        let _ = release_lock(&lock_path_for_destination(&self.config.destination)).await;
    }
}

fn file_size(path: &Path) -> Option<u64> {
    path.metadata().ok().map(|metadata| metadata.len())
}

fn remove_file(path: &Path) {
    let _ = std::fs::remove_file(path);
}

fn remove_resume_artifact(destination: &Path) {
    remove_file(&destination.with_extension("part"));
    remove_file(&destination.with_extension("resume_data"));
}

fn lock_path_for_destination(destination: &Path) -> PathBuf {
    PathBuf::from(format!("{}.lock", destination.display()))
}

fn crc_path_for_destination(destination: &Path) -> Option<PathBuf> {
    let crc_path = crc_path_for_file(destination);
    crc_path.exists().then_some(crc_path)
}

async fn validate_completed_file(config: &DownloadConfig) -> Result<u64, String> {
    let mut metadata = config.destination.metadata().ok();
    for _attempt in 0..10 {
        if metadata.is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        metadata = config.destination.metadata().ok();
    }
    let metadata = metadata.ok_or_else(|| "download completed but destination is missing".to_string())?;
    let total_bytes = config.expected_bytes.unwrap_or(metadata.len());

    match &config.file_check {
        FileCheck::None => Ok(total_bytes),
        FileCheck::CRC(expected_crc) => match calculate_and_verify_crc(&config.destination, expected_crc) {
            Ok(true) => {
                let _ = save_crc_file(&config.destination, expected_crc);
                Ok(total_bytes)
            },
            Ok(false) => Err("CRC verification failed".to_string()),
            Err(error) => Err(format!("CRC verification error: {error}")),
        },
    }
}
