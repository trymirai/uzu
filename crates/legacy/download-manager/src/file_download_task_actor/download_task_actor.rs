use std::{
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use kiban::fs;
use tokio::sync::{
    Mutex as TokioMutex,
    broadcast::Sender as TokioBroadcastSender,
    mpsc::Receiver as TokioMpscReceiver,
    oneshot::Sender as TokioOneshotSender,
    watch::{Receiver as TokioWatchReceiver, Sender as TokioWatchSender},
};

use crate::{
    DownloadError, FileCheck, FileDownloadState, LockFileState, check_lock_file,
    crc_utils::{calculate_and_verify_crc, crc_path_for_file, save_crc_file},
    download_log_event::{DownloadLogEvent, log},
    file_download_task_actor::{
        BackendEvent, DownloadActorState, PendingProgressSlot, ProgressCounters, PublicProjection, TaskCommand,
        TerminalOutcome, project_runtime_public_state,
    },
    lock_manager::{DestinationLockLease, lock_path_for_destination},
    release_lock_if_owned,
    traits::{
        ActiveDownloadGeneration, ActiveDownloadGenerationCounter, ActiveTask, BackendContext, BackendEventSender,
        DownloadBackend, DownloadConfig,
    },
};

enum ActorLoopExit {
    AlreadyStopped,
    PreserveArtifacts,
}

pub struct DownloadTaskActor<B: DownloadBackend> {
    config: Arc<DownloadConfig>,
    context: Arc<B::Context>,
    backend_event_sender: BackendEventSender,
    generation_counter: ActiveDownloadGenerationCounter,
    state: DownloadActorState<B>,
    projection: PublicProjection,
    progress_counters: ProgressCounters,
    command_receiver: TokioMpscReceiver<TaskCommand>,
    backend_event_receiver: TokioMpscReceiver<BackendEvent>,
    pending_progress: Arc<TokioMutex<PendingProgressSlot>>,
    progress_waker_receiver: TokioWatchReceiver<()>,
    public_state_sender: TokioWatchSender<FileDownloadState>,
    progress_sender: TokioBroadcastSender<FileDownloadState>,
    terminal_sender: TokioWatchSender<TerminalOutcome>,
    pending_terminal_outcome: Option<TerminalOutcome>,
}

impl<B: DownloadBackend> DownloadTaskActor<B> {
    pub fn new(
        config: Arc<DownloadConfig>,
        context: Arc<B::Context>,
        backend_event_sender: BackendEventSender,
        generation_counter: ActiveDownloadGenerationCounter,
        state: DownloadActorState<B>,
        projection: PublicProjection,
        progress_counters: ProgressCounters,
        command_receiver: TokioMpscReceiver<TaskCommand>,
        backend_event_receiver: TokioMpscReceiver<BackendEvent>,
        pending_progress: Arc<TokioMutex<PendingProgressSlot>>,
        progress_waker_receiver: TokioWatchReceiver<()>,
        public_state_sender: TokioWatchSender<FileDownloadState>,
        progress_sender: TokioBroadcastSender<FileDownloadState>,
        terminal_sender: TokioWatchSender<TerminalOutcome>,
    ) -> Self {
        Self {
            config,
            context,
            backend_event_sender,
            generation_counter,
            state,
            projection,
            progress_counters,
            command_receiver,
            backend_event_receiver,
            pending_progress,
            progress_waker_receiver,
            public_state_sender,
            progress_sender,
            terminal_sender,
            pending_terminal_outcome: None,
        }
    }

    pub async fn run(mut self) {
        self.publish_current_state();

        let mut loop_exit = ActorLoopExit::PreserveArtifacts;
        loop {
            tokio::select! {
                command = self.command_receiver.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    if !self.handle_command(command).await {
                        loop_exit = ActorLoopExit::AlreadyStopped;
                        break;
                    }
                }
                backend_event = self.backend_event_receiver.recv() => {
                    let Some(backend_event) = backend_event else {
                        break;
                    };
                    self.handle_backend_event(backend_event).await;
                    self.publish_current_state();
                    self.flush_terminal_outcome();
                }
                progress_wake_result = self.progress_waker_receiver.changed() => {
                    if progress_wake_result.is_err() {
                        break;
                    }
                    self.handle_pending_progress().await;
                    self.publish_current_state();
                    self.flush_terminal_outcome();
                }
            }
        }

        if matches!(loop_exit, ActorLoopExit::PreserveArtifacts) {
            self.stop_preserving_artifacts().await;
            self.publish_current_state();
            self.flush_terminal_outcome();
        }
        let _ = self.terminal_sender.send(TerminalOutcome::ActorStopped);
    }

    async fn stop_preserving_artifacts(&mut self) {
        let current_state = std::mem::replace(&mut self.state, DownloadActorState::NotDownloaded);
        let from_state = current_state.name();
        match current_state {
            DownloadActorState::Downloading {
                active_task,
                destination_lease,
                ..
            } => match active_task.pause(&self.config.destination).await {
                Ok(part_path) => {
                    let downloaded_bytes = B::read_resume_progress(&part_path).await.unwrap_or(0);
                    self.progress_counters = ProgressCounters {
                        downloaded_bytes,
                        total_bytes: self.config.expected_bytes.unwrap_or(downloaded_bytes),
                    };
                    release_destination_lease(destination_lease).await;
                    self.finish_transition(
                        from_state,
                        DownloadActorState::Paused {
                            part_path,
                        },
                    );
                },
                Err(error) => {
                    tracing::debug!("failed to preserve resume data while stopping download actor: {error}");
                    self.progress_counters = ProgressCounters::default();
                    release_destination_lease(destination_lease).await;
                    self.finish_transition(from_state, DownloadActorState::NotDownloaded);
                },
            },
            state => {
                self.state = state;
            },
        }

        let lock_path = lock_path_for_destination(&self.config.destination);
        let _ = release_lock_if_owned(&lock_path, &self.config.manager_id, self.config.manager_instance_id).await;
    }

    async fn handle_command(
        &mut self,
        command: TaskCommand,
    ) -> bool {
        match command {
            TaskCommand::Download {
                reply_sender,
            } => {
                let result = self.handle_download().await;
                self.publish_current_state();
                self.flush_terminal_outcome();
                send_reply(reply_sender, result);
                true
            },
            TaskCommand::Pause {
                reply_sender,
            } => {
                let result = self.handle_pause().await;
                self.publish_current_state();
                self.flush_terminal_outcome();
                send_reply(reply_sender, result);
                true
            },
            TaskCommand::Cancel {
                reply_sender,
            } => {
                let result = self.handle_cancel_or_remove().await;
                self.publish_current_state();
                self.flush_terminal_outcome();
                send_reply(reply_sender, result);
                true
            },
            TaskCommand::Remove {
                reply_sender,
            } => {
                let result = self.handle_cancel_or_remove().await;
                self.publish_current_state();
                self.flush_terminal_outcome();
                send_reply(reply_sender, result);
                false
            },
        }
    }

    async fn handle_download(&mut self) -> Result<(), DownloadError> {
        match &self.state {
            DownloadActorState::NotDownloaded => self.start_fresh_download().await,
            DownloadActorState::Paused {
                part_path,
            } => self.resume_download(part_path.clone()).await,
            DownloadActorState::Downloading {
                ..
            }
            | DownloadActorState::Downloaded => Ok(()),
        }
    }

    async fn handle_pause(&mut self) -> Result<(), DownloadError> {
        let current_state = std::mem::replace(&mut self.state, DownloadActorState::NotDownloaded);
        let from_state = current_state.name();
        match current_state {
            DownloadActorState::Downloading {
                active_task,
                destination_lease,
                ..
            } => {
                let pause_result = active_task.pause(&self.config.destination).await;
                release_destination_lease(destination_lease).await;
                match pause_result {
                    Ok(part_path) => {
                        let downloaded_bytes = B::read_resume_progress(&part_path).await.unwrap_or(0);
                        self.progress_counters = ProgressCounters {
                            downloaded_bytes,
                            total_bytes: self.config.expected_bytes.unwrap_or(downloaded_bytes),
                        };
                        self.finish_transition(
                            from_state,
                            DownloadActorState::Paused {
                                part_path,
                            },
                        );
                        Ok(())
                    },
                    Err(error) => {
                        let message = error.to_string();
                        self.projection = PublicProjection::StickyError(message.clone());
                        self.progress_counters = ProgressCounters::default();
                        self.pending_terminal_outcome = Some(TerminalOutcome::Error(message.clone()));
                        self.finish_transition(from_state, DownloadActorState::NotDownloaded);
                        Err(DownloadError::Backend(message))
                    },
                }
            },
            DownloadActorState::Paused {
                ..
            } => {
                self.state = current_state;
                Ok(())
            },
            state => {
                self.state = state;
                Err(DownloadError::InvalidStateTransition)
            },
        }
    }

    async fn handle_cancel_or_remove(&mut self) -> Result<(), DownloadError> {
        let current_state = std::mem::replace(&mut self.state, DownloadActorState::NotDownloaded);
        let from_state = current_state.name();
        match current_state {
            DownloadActorState::Downloading {
                active_task,
                destination_lease,
                ..
            } => {
                let _ = active_task.cancel(&self.config.destination).await;
                remove_resume_artifact(&self.config.destination).await;
                self.progress_counters = ProgressCounters::default();
                self.projection = PublicProjection::None;
                release_destination_lease(destination_lease).await;
                self.finish_transition(from_state, DownloadActorState::NotDownloaded);
                Ok(())
            },
            DownloadActorState::Paused {
                part_path,
            } => {
                remove_file(&part_path).await;
                self.progress_counters = ProgressCounters::default();
                self.projection = PublicProjection::None;
                self.finish_transition(from_state, DownloadActorState::NotDownloaded);
                Ok(())
            },
            state => {
                self.state = state;
                if matches!(self.projection, PublicProjection::StickyError(_)) {
                    self.projection = PublicProjection::None;
                    self.progress_counters = ProgressCounters::default();
                }
                Ok(())
            },
        }
    }

    async fn handle_backend_event(
        &mut self,
        backend_event: BackendEvent,
    ) {
        match backend_event {
            BackendEvent::Completed {
                generation,
            } => self.handle_backend_completed(generation).await,
            BackendEvent::Error {
                generation,
                message,
            } => self.handle_backend_error(generation, message).await,
        }
    }

    async fn handle_backend_completed(
        &mut self,
        completed_generation: ActiveDownloadGeneration,
    ) {
        let should_handle = matches!(
            &self.state,
            DownloadActorState::Downloading {
                generation,
                ..
            } if *generation == completed_generation
        );
        if !should_handle {
            return;
        }

        let current_state = std::mem::replace(&mut self.state, DownloadActorState::NotDownloaded);
        let from_state = current_state.name();
        if let DownloadActorState::Downloading {
            destination_lease,
            ..
        } = current_state
        {
            match validate_completed_file(&self.config).await {
                Ok(total_bytes) => {
                    remove_resume_artifact(&self.config.destination).await;
                    self.progress_counters = ProgressCounters {
                        downloaded_bytes: total_bytes,
                        total_bytes,
                    };
                    self.projection = PublicProjection::None;
                    release_destination_lease(destination_lease).await;
                    self.finish_transition(from_state, DownloadActorState::Downloaded);
                    self.pending_terminal_outcome = Some(TerminalOutcome::Downloaded);
                },
                Err(message) => {
                    remove_file(&self.config.destination).await;
                    remove_resume_artifact(&self.config.destination).await;
                    remove_file(&crc_path_for_file(&self.config.destination)).await;
                    self.progress_counters = ProgressCounters::default();
                    self.projection = PublicProjection::StickyError(message.clone());
                    release_destination_lease(destination_lease).await;
                    self.finish_transition(from_state, DownloadActorState::NotDownloaded);
                    self.pending_terminal_outcome = Some(TerminalOutcome::Error(message));
                },
            }
        }
    }

    async fn handle_backend_error(
        &mut self,
        error_generation: ActiveDownloadGeneration,
        message: String,
    ) {
        let should_handle = matches!(
            &self.state,
            DownloadActorState::Downloading {
                generation,
                ..
            } if *generation == error_generation
        );
        if !should_handle {
            return;
        }

        let current_state = std::mem::replace(&mut self.state, DownloadActorState::NotDownloaded);
        let from_state = current_state.name();
        if let DownloadActorState::Downloading {
            active_task,
            destination_lease,
            ..
        } = current_state
        {
            let _ = active_task.cancel(&self.config.destination).await;
            remove_resume_artifact(&self.config.destination).await;
            self.projection = PublicProjection::StickyError(message.clone());
            self.progress_counters = ProgressCounters::default();
            release_destination_lease(destination_lease).await;
            self.finish_transition(from_state, DownloadActorState::NotDownloaded);
            self.pending_terminal_outcome = Some(TerminalOutcome::Error(message));
        }
    }

    async fn handle_pending_progress(&mut self) {
        let progress = self.pending_progress.lock().await.take();
        let Some(progress) = progress else {
            return;
        };

        if let DownloadActorState::Downloading {
            generation,
            ..
        } = &self.state
            && *generation == progress.generation
        {
            self.progress_counters = ProgressCounters {
                downloaded_bytes: progress.downloaded_bytes,
                total_bytes: progress.total_bytes.or(self.config.expected_bytes).unwrap_or(progress.downloaded_bytes),
            };
        }
    }

    async fn start_fresh_download(&mut self) -> Result<(), DownloadError> {
        let lease = self.acquire_destination_lease().await?;
        let generation = self.generation_counter.allocate_next();
        let active_task = match self
            .context
            .download(Arc::clone(&self.config), generation, self.backend_event_sender.clone(), &lease)
            .await
        {
            Ok(active_task) => active_task,
            Err(error) => {
                release_destination_lease(lease).await;
                return Err(DownloadError::Backend(error.to_string()));
            },
        };

        self.projection = PublicProjection::None;
        self.progress_counters = ProgressCounters {
            downloaded_bytes: 0,
            total_bytes: self.config.expected_bytes.unwrap_or(0),
        };
        self.transition_to(DownloadActorState::Downloading {
            active_task,
            generation,
            destination_lease: lease,
        });
        Ok(())
    }

    async fn resume_download(
        &mut self,
        part_path: PathBuf,
    ) -> Result<(), DownloadError> {
        if !fs::asyn::try_exists(&part_path).await.unwrap_or(false) {
            remove_file(&part_path).await;
            return self.start_fresh_download().await;
        }

        let lease = self.acquire_destination_lease().await?;
        let generation = self.generation_counter.allocate_next();
        let resume_bytes = B::read_resume_progress(&part_path).await.unwrap_or(0);
        let active_task = match self
            .context
            .resume(Arc::clone(&self.config), generation, &part_path, self.backend_event_sender.clone(), &lease)
            .await
        {
            Ok(active_task) => active_task,
            Err(error) => {
                release_destination_lease(lease).await;
                remove_file(&part_path).await;
                let message = error.to_string();
                self.projection = PublicProjection::StickyError(message.clone());
                self.progress_counters = ProgressCounters::default();
                self.transition_to(DownloadActorState::NotDownloaded);
                self.pending_terminal_outcome = Some(TerminalOutcome::Error(message.clone()));
                return Err(DownloadError::Backend(message));
            },
        };

        self.projection = PublicProjection::None;
        self.progress_counters = ProgressCounters {
            downloaded_bytes: resume_bytes,
            total_bytes: self.config.expected_bytes.unwrap_or(resume_bytes),
        };
        self.transition_to(DownloadActorState::Downloading {
            active_task,
            generation,
            destination_lease: lease,
        });
        Ok(())
    }

    async fn acquire_destination_lease(&mut self) -> Result<DestinationLockLease, DownloadError> {
        let lock_path = lock_path_for_destination(&self.config.destination);
        match check_lock_file(
            &lock_path,
            &self.config.manager_id,
            self.config.manager_instance_id,
            kiban::process::id(),
        )
        .await
        {
            LockFileState::OwnedByOtherApp(lock_file_info) => {
                self.projection = PublicProjection::LockedByOther(lock_file_info.manager_id.clone());
                Err(DownloadError::LockedByOther(lock_file_info.manager_id))
            },
            LockFileState::Missing
            | LockFileState::OwnedByUs(_)
            | LockFileState::OwnedBySameAppOldProcess(_)
            | LockFileState::Stale(_)
            | LockFileState::StaleUnparseable(_) => {
                match DestinationLockLease::acquire(
                    &lock_path,
                    &self.config.manager_id,
                    self.config.manager_instance_id,
                )
                .await
                {
                    Ok(lease) => Ok(lease),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        match check_lock_file(
                            &lock_path,
                            &self.config.manager_id,
                            self.config.manager_instance_id,
                            kiban::process::id(),
                        )
                        .await
                        {
                            LockFileState::OwnedByOtherApp(lock_file_info) => {
                                self.projection = PublicProjection::LockedByOther(lock_file_info.manager_id.clone());
                                Err(DownloadError::LockedByOther(lock_file_info.manager_id))
                            },
                            _ => Err(DownloadError::from(error)),
                        }
                    },
                    Err(error) => Err(DownloadError::from(error)),
                }
            },
        }
    }

    fn publish_current_state(&self) {
        let public_state =
            project_runtime_public_state(&self.state, &self.projection, self.progress_counters, &self.config);
        let _ = self.public_state_sender.send(public_state.clone());
        let _ = self.progress_sender.send(public_state);
    }

    fn flush_terminal_outcome(&mut self) {
        if let Some(terminal_outcome) = self.pending_terminal_outcome.take() {
            let _ = self.terminal_sender.send(terminal_outcome);
        }
    }

    fn transition_to(
        &mut self,
        next_state: DownloadActorState<B>,
    ) {
        let from_state = self.state.name();
        self.finish_transition(from_state, next_state);
    }

    fn finish_transition(
        &mut self,
        from_state: &'static str,
        next_state: DownloadActorState<B>,
    ) {
        let to_state = next_state.name();
        self.state = next_state;
        if from_state != to_state {
            log(DownloadLogEvent::StateTransition {
                download_id: self.config.download_id,
                from: from_state,
                to: to_state,
            });
        }
    }
}

fn send_reply(
    reply_sender: TokioOneshotSender<Result<(), DownloadError>>,
    result: Result<(), DownloadError>,
) {
    let _ = reply_sender.send(result);
}

async fn release_destination_lease(destination_lease: DestinationLockLease) {
    let _ = destination_lease.release().await;
}

async fn remove_file(path: &Path) {
    let _ = fs::asyn::remove_file(path).await;
}

async fn remove_resume_artifact(destination: &Path) {
    remove_file(&destination.with_extension("part")).await;
    remove_file(&destination.with_extension("resume_data")).await;
}

async fn validate_completed_file(config: &DownloadConfig) -> Result<u64, String> {
    // After the backend reports completion the destination may not yet be fully
    // visible on disk: metadata can lag, and on the copy/move fallback path the
    // file briefly exists with fewer bytes than expected. Retry until it is
    // present with the expected size (or the budget runs out) so a transient
    // mismatch is not mistaken for a corrupt download. A genuinely truncated file
    // has a stable size and still fails once the retries are exhausted.
    for _ in 0..10 {
        let is_ready = match fs::asyn::file_length(config.destination.as_path()).await {
            Ok(dst_len) => match config.expected_bytes {
                Some(expected_bytes) => dst_len == expected_bytes,
                None => true,
            },
            Err(err) if err.kind() == ErrorKind::NotFound => false,
            Err(err) => return Err(err.to_string()),
        };
        if is_ready {
            break;
        }

        kiban::time::sleep(Duration::from_millis(50)).await;
    }

    let actual_bytes = fs::asyn::file_length(config.destination.as_path()).await.map_err(|err| err.to_string())?;
    if let Some(expected_bytes) = config.expected_bytes
        && expected_bytes != actual_bytes
    {
        return Err(format!("downloaded file is {actual_bytes} bytes but registry declared {expected_bytes}"));
    }

    let total_bytes = config.expected_bytes.unwrap_or(actual_bytes);

    match &config.file_check {
        FileCheck::None => Ok(total_bytes),
        FileCheck::CRC(expected_crc) => {
            let crc_result = calculate_and_verify_crc(&config.destination, expected_crc).await;
            match crc_result {
                Ok(true) => {
                    let destination = config.destination.clone();
                    let expected = expected_crc.clone();
                    let _ = save_crc_file(&destination, &expected).await;
                    Ok(total_bytes)
                },
                Ok(false) => Err("CRC verification failed".to_string()),
                Err(error) => Err(format!("CRC verification error: {error}")),
            }
        },
    }
}
