use std::{
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use kiban::fs;
use tokio::sync::{
    Mutex as TokioMutex,
    mpsc::Receiver as TokioMpscReceiver,
    oneshot::Sender as TokioOneshotSender,
    watch::{Receiver as TokioWatchReceiver, Sender as TokioWatchSender},
};

use crate::{
    DownloadCleanupFailure, DownloadError, FileDownloadSnapshot, LockFileState,
    backends::common::reject_symlink_components,
    check_lock_file,
    crc_utils::{VerificationError, VerificationStatus, save_integrity_cache_at, verify_file_integrity},
    download_log_event::{DownloadLogEvent, log},
    file_download_task::InactiveTaskShutdown,
    file_download_task_actor::{
        BackendEvent, BackendProgress, DownloadActorState, ProgressCounters, PublicProjection, TaskCommand,
        project_runtime_public_state,
    },
    lock_manager::DestinationLockLease,
    release_lock_if_owned,
    traits::{
        ActiveDownloadGeneration, ActiveDownloadGenerationCounter, ActiveTask, ActiveTaskPauseOutcome, BackendContext,
        BackendEventSender, DownloadBackend, DownloadConfig,
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
    pending_progress: Arc<TokioMutex<Option<BackendProgress>>>,
    progress_waker_receiver: TokioWatchReceiver<()>,
    snapshot_sender: TokioWatchSender<FileDownloadSnapshot>,
    transfer_retry_count: u16,
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
        pending_progress: Arc<TokioMutex<Option<BackendProgress>>>,
        progress_waker_receiver: TokioWatchReceiver<()>,
        snapshot_sender: TokioWatchSender<FileDownloadSnapshot>,
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
            snapshot_sender,
            transfer_retry_count: 0,
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
                }
                progress_wake_result = self.progress_waker_receiver.changed() => {
                    if progress_wake_result.is_err() {
                        break;
                    }
                    self.handle_pending_progress().await;
                    self.publish_current_state();
                }
            }
        }

        if matches!(loop_exit, ActorLoopExit::PreserveArtifacts) {
            self.stop_preserving_artifacts().await;
            self.publish_current_state();
        }
    }

    async fn stop_preserving_artifacts(&mut self) {
        let current_state = std::mem::replace(&mut self.state, DownloadActorState::NotDownloaded);
        let from_state = current_state.name();
        match current_state {
            DownloadActorState::Downloading {
                active_task,
                destination_lease,
                ..
            } => {
                if let Err(failure) = validate_destructive_cleanup_paths(&self.config).await {
                    active_task.cancel(&self.config.destination).await;
                    self.projection = PublicProjection::StickyError(DownloadError::cleanup_failures(vec![failure]));
                    self.reset_downloaded_bytes();
                    release_destination_lease(destination_lease).await;
                    self.finish_transition(from_state, DownloadActorState::NotDownloaded);
                } else {
                    match active_task.pause(&self.config.destination).await {
                        Ok(ActiveTaskPauseOutcome::Paused(part_path)) => {
                            let downloaded_bytes = B::read_resume_progress(&part_path).await.unwrap_or(0);
                            let total_bytes = self.progress_counters.total_bytes.or(self.config.expected_bytes);
                            self.progress_counters = ProgressCounters {
                                downloaded_bytes,
                                total_bytes,
                            };
                            release_destination_lease(destination_lease).await;
                            self.finish_transition(
                                from_state,
                                DownloadActorState::Paused {
                                    part_path,
                                },
                            );
                        },
                        Ok(ActiveTaskPauseOutcome::Completed) => {
                            self.finish_completed_download(from_state, destination_lease).await;
                        },
                        Ok(ActiveTaskPauseOutcome::Failed(error)) => {
                            self.finish_failed_download(from_state, destination_lease, error).await;
                        },
                        Err(error) => {
                            tracing::debug!("failed to preserve resume data while stopping download actor: {error}");
                            self.reset_downloaded_bytes();
                            release_destination_lease(destination_lease).await;
                            self.finish_transition(from_state, DownloadActorState::NotDownloaded);
                        },
                    }
                }
            },
            state => {
                self.state = state;
            },
        }

        let lock_path = self.config.lock_path();
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
                if let Err(error) = &result {
                    self.record_download_start_failure(error);
                }
                self.publish_current_state();
                send_reply(reply_sender, result);
                true
            },
            TaskCommand::Pause {
                reply_sender,
            } => {
                let result = self.handle_pause().await;
                self.publish_current_state();
                send_reply(reply_sender, result);
                true
            },
            TaskCommand::Cancel {
                reply_sender,
            } => {
                let result = self.handle_cancel_or_remove().await;
                self.publish_current_state();
                send_reply(reply_sender, result);
                true
            },
            TaskCommand::CancelAndDelete {
                reply_sender,
            } => {
                let result = self.handle_cancel_and_delete().await;
                self.publish_current_state();
                send_reply(reply_sender, result);
                true
            },
            TaskCommand::Remove {
                reply_sender,
            } => {
                let result = self.handle_cancel_or_remove().await;
                self.publish_current_state();
                send_reply(reply_sender, result);
                false
            },
            TaskCommand::RemoveIfInactive {
                reply_sender,
            } => {
                let result = self.handle_remove_if_inactive().await;
                let should_stop = matches!(result, Ok(InactiveTaskShutdown::Stopped));
                self.publish_current_state();
                let _ = reply_sender.send(result);
                !should_stop
            },
            TaskCommand::StopPreservingArtifactsIfInactive {
                reply_sender,
            } => {
                let result = if matches!(self.state, DownloadActorState::Downloading { .. }) {
                    InactiveTaskShutdown::Active
                } else {
                    InactiveTaskShutdown::Stopped
                };
                let _ = reply_sender.send(Ok(result));
                result == InactiveTaskShutdown::Active
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

    fn record_download_start_failure(
        &mut self,
        error: &DownloadError,
    ) {
        if !matches!(error, DownloadError::LockedByOther(_)) {
            self.projection = PublicProjection::StickyError(error.clone());
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
                if let Err(failure) = validate_destructive_cleanup_paths(&self.config).await {
                    active_task.cancel(&self.config.destination).await;
                    let error = DownloadError::cleanup_failures(vec![failure]);
                    self.projection = PublicProjection::StickyError(error.clone());
                    self.reset_downloaded_bytes();
                    release_destination_lease(destination_lease).await;
                    self.finish_transition(from_state, DownloadActorState::NotDownloaded);
                    return Err(error);
                }
                let pause_result = active_task.pause(&self.config.destination).await;
                match pause_result {
                    Ok(ActiveTaskPauseOutcome::Paused(part_path)) => {
                        let downloaded_bytes = B::read_resume_progress(&part_path).await.unwrap_or(0);
                        let total_bytes = self.progress_counters.total_bytes.or(self.config.expected_bytes);
                        self.progress_counters = ProgressCounters {
                            downloaded_bytes,
                            total_bytes,
                        };
                        release_destination_lease(destination_lease).await;
                        self.finish_transition(
                            from_state,
                            DownloadActorState::Paused {
                                part_path,
                            },
                        );
                        Ok(())
                    },
                    Ok(ActiveTaskPauseOutcome::Completed) => {
                        self.finish_completed_download(from_state, destination_lease).await;
                        Ok(())
                    },
                    Ok(ActiveTaskPauseOutcome::Failed(error)) => {
                        self.finish_failed_download(from_state, destination_lease, error.clone()).await;
                        Err(error)
                    },
                    Err(error) => {
                        let error = DownloadError::Backend(error.to_string());
                        self.projection = PublicProjection::StickyError(error.clone());
                        self.reset_downloaded_bytes();
                        release_destination_lease(destination_lease).await;
                        self.finish_transition(from_state, DownloadActorState::NotDownloaded);
                        Err(error)
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
                active_task.cancel(&self.config.destination).await;
                remove_resume_artifacts(&self.config).await;
                self.reset_downloaded_bytes();
                self.projection = PublicProjection::None;
                release_destination_lease(destination_lease).await;
                self.finish_transition(from_state, DownloadActorState::NotDownloaded);
                Ok(())
            },
            DownloadActorState::Paused {
                ..
            } => {
                remove_resume_artifacts(&self.config).await;
                self.reset_downloaded_bytes();
                self.projection = PublicProjection::None;
                self.finish_transition(from_state, DownloadActorState::NotDownloaded);
                Ok(())
            },
            state => {
                self.state = state;
                if matches!(self.projection, PublicProjection::StickyError(_)) {
                    self.projection = PublicProjection::None;
                    self.reset_downloaded_bytes();
                }
                Ok(())
            },
        }
    }

    async fn handle_remove_if_inactive(&mut self) -> Result<InactiveTaskShutdown, DownloadError> {
        match &self.state {
            DownloadActorState::Downloading {
                ..
            } => return Ok(InactiveTaskShutdown::Active),
            DownloadActorState::Paused {
                part_path,
            } => {
                let part_path = part_path.clone();
                let destination_lease = match self.acquire_destination_lease().await {
                    Ok(destination_lease) => destination_lease,
                    Err(DownloadError::LockedByOther(_)) => return Ok(InactiveTaskShutdown::Active),
                    Err(error) => return Err(error),
                };
                let cleanup_result = async {
                    validate_destructive_cleanup_paths(&self.config)
                        .await
                        .map_err(|failure| DownloadError::cleanup_failures(vec![failure]))?;
                    remove_file_if_exists(&part_path).await?;
                    remove_file_if_exists(&self.config.recovery_metadata_path()).await?;
                    remove_file_if_exists(&self.config.recovery_metadata_staging_path()).await?;
                    Ok::<(), DownloadError>(())
                }
                .await;
                release_destination_lease(destination_lease).await;
                cleanup_result?;

                let from_state = self.state.name();
                self.reset_downloaded_bytes();
                self.projection = PublicProjection::None;
                self.finish_transition(from_state, DownloadActorState::NotDownloaded);
            },
            DownloadActorState::NotDownloaded | DownloadActorState::Downloaded => {},
        }

        Ok(InactiveTaskShutdown::Stopped)
    }

    async fn handle_cancel_and_delete(&mut self) -> Result<(), DownloadError> {
        let current_state = std::mem::replace(&mut self.state, DownloadActorState::NotDownloaded);
        let from_state = current_state.name();
        let destination_lease = match current_state {
            DownloadActorState::Downloading {
                active_task,
                destination_lease,
                ..
            } => {
                active_task.cancel(&self.config.destination).await;
                destination_lease
            },
            DownloadActorState::Paused {
                part_path,
            } => {
                self.state = DownloadActorState::Paused {
                    part_path: part_path.clone(),
                };
                let destination_lease = self.acquire_destination_lease().await?;
                let _ = std::mem::replace(&mut self.state, DownloadActorState::NotDownloaded);
                destination_lease
            },
            state => {
                self.state = state;
                let destination_lease = self.acquire_destination_lease().await?;
                let _ = std::mem::replace(&mut self.state, DownloadActorState::NotDownloaded);
                destination_lease
            },
        };

        let mut cleanup_failures = remove_download_files(&self.config).await;
        let lock_path = self.config.lock_path();
        if let Err(error) = destination_lease.release().await {
            cleanup_failures.push(DownloadCleanupFailure::new(&lock_path, &error));
        }
        if cleanup_failures.is_empty() {
            match validate_destructive_cleanup_paths(&self.config).await {
                Ok(()) => {
                    if let Err(error) = remove_directory_if_empty(&self.config.artifact_root).await {
                        cleanup_failures.push(DownloadCleanupFailure::new(&self.config.artifact_root, &error));
                    }
                },
                Err(failure) => cleanup_failures.push(failure),
            }
        }
        let cleanup_result = if cleanup_failures.is_empty() {
            Ok(())
        } else {
            Err(DownloadError::cleanup_failures(cleanup_failures))
        };

        match cleanup_result {
            Ok(()) => {
                self.reset_downloaded_bytes();
                self.projection = PublicProjection::None;
                self.finish_transition(from_state, DownloadActorState::NotDownloaded);
                Ok(())
            },
            Err(error) => {
                self.projection = PublicProjection::StickyError(error.clone());
                self.finish_transition(from_state, DownloadActorState::NotDownloaded);
                Err(error)
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
                error,
            } => self.handle_backend_error(generation, error).await,
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
            self.finish_completed_download(from_state, destination_lease).await;
        }
    }

    async fn handle_backend_error(
        &mut self,
        error_generation: ActiveDownloadGeneration,
        error: DownloadError,
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
            active_task.cancel(&self.config.destination).await;
            if error.is_retryable_transfer_failure() && self.transfer_retry_count < B::TERMINAL_RETRY_COUNT {
                self.transfer_retry_count = self.transfer_retry_count.saturating_add(1);
                kiban::time::sleep(Duration::from_millis(250)).await;
                let generation = self.generation_counter.allocate_next();
                match self
                    .context
                    .download(
                        Arc::clone(&self.config),
                        generation,
                        self.backend_event_sender.clone(),
                        &destination_lease,
                    )
                    .await
                {
                    Ok(active_task) => {
                        tracing::debug!(
                            retry = self.transfer_retry_count,
                            error = %error,
                            "retrying backend download after a transient terminal failure"
                        );
                        self.projection = PublicProjection::None;
                        self.progress_counters.downloaded_bytes = 0;
                        self.finish_transition(
                            from_state,
                            DownloadActorState::Downloading {
                                active_task,
                                generation,
                                destination_lease,
                            },
                        );
                        return;
                    },
                    Err(start_error) => {
                        self.finish_failed_download(
                            from_state,
                            destination_lease,
                            DownloadError::Backend(start_error.to_string()),
                        )
                        .await;
                        return;
                    },
                }
            }
            self.finish_failed_download(from_state, destination_lease, error).await;
        }
    }

    async fn finish_completed_download(
        &mut self,
        from_state: &'static str,
        destination_lease: DestinationLockLease,
    ) {
        match validate_completed_file(&self.config).await {
            Ok(total_bytes) => {
                remove_resume_artifacts(&self.config).await;
                self.progress_counters = ProgressCounters {
                    downloaded_bytes: total_bytes,
                    total_bytes: Some(total_bytes),
                };
                self.projection = PublicProjection::None;
                release_destination_lease(destination_lease).await;
                self.finish_transition(from_state, DownloadActorState::Downloaded);
            },
            Err(error) => {
                if error.destination_is_invalid() {
                    remove_owned_file(
                        &self.config.destination,
                        self.config.destination.parent(),
                        "invalid destination",
                    )
                    .await;
                    remove_owned_file(
                        &self.config.integrity_receipt_path(),
                        Some(&self.config.artifact_root),
                        "invalid integrity receipt",
                    )
                    .await;
                }
                let error = error.into_download_error(&self.config.destination);
                self.finish_failed_download(from_state, destination_lease, error).await;
            },
        }
    }

    async fn finish_failed_download(
        &mut self,
        from_state: &'static str,
        destination_lease: DestinationLockLease,
        error: DownloadError,
    ) {
        remove_resume_artifacts(&self.config).await;
        self.projection = PublicProjection::StickyError(error.clone());
        self.reset_downloaded_bytes();
        release_destination_lease(destination_lease).await;
        self.finish_transition(from_state, DownloadActorState::NotDownloaded);
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
                total_bytes: progress.total_bytes.or(self.config.expected_bytes),
            };
        }
    }

    async fn start_fresh_download(&mut self) -> Result<(), DownloadError> {
        self.transfer_retry_count = 0;
        let lease = self.acquire_destination_lease().await?;
        let generation = self.generation_counter.allocate_next();
        let active_task = match self
            .context
            .download(Arc::clone(&self.config), generation, self.backend_event_sender.clone(), &lease)
            .await
        {
            Ok(active_task) => active_task,
            Err(error) => {
                remove_resume_artifacts(&self.config).await;
                release_destination_lease(lease).await;
                return Err(DownloadError::Backend(error.to_string()));
            },
        };

        self.projection = PublicProjection::None;
        self.progress_counters = ProgressCounters {
            downloaded_bytes: 0,
            total_bytes: self.config.expected_bytes,
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
        self.transfer_retry_count = 0;
        if !fs::asyn::try_exists(&part_path).await.unwrap_or(false) {
            remove_owned_file(&part_path, Some(&self.config.artifact_root), "stale resume artifact").await;
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
                remove_resume_artifacts(&self.config).await;
                release_destination_lease(lease).await;
                let error = DownloadError::Backend(error.to_string());
                self.projection = PublicProjection::StickyError(error.clone());
                self.reset_downloaded_bytes();
                self.transition_to(DownloadActorState::NotDownloaded);
                return Err(error);
            },
        };

        self.projection = PublicProjection::None;
        let total_bytes = self.progress_counters.total_bytes.or(self.config.expected_bytes);
        self.progress_counters = ProgressCounters {
            downloaded_bytes: resume_bytes,
            total_bytes,
        };
        self.transition_to(DownloadActorState::Downloading {
            active_task,
            generation,
            destination_lease: lease,
        });
        Ok(())
    }

    async fn acquire_destination_lease(&mut self) -> Result<DestinationLockLease, DownloadError> {
        let lock_path = self.config.lock_path();
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
        self.snapshot_sender.send_replace(FileDownloadSnapshot::with_total_bytes(
            public_state,
            self.projection.failure(),
            self.progress_counters.total_bytes.or(self.config.expected_bytes),
        ));
    }

    fn reset_downloaded_bytes(&mut self) {
        self.progress_counters = ProgressCounters {
            downloaded_bytes: 0,
            total_bytes: self.progress_counters.total_bytes.or(self.config.expected_bytes),
        };
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

async fn remove_resume_artifacts(config: &DownloadConfig) {
    let artifact_root = Some(config.artifact_root.as_path());
    let artifacts = [
        config.resume_artifact_path("part"),
        config.resume_artifact_path("resume_data"),
        config.installation_artifact_path(),
        config.recovery_metadata_path(),
        config.recovery_metadata_staging_path(),
    ];
    for artifact in artifacts {
        remove_owned_file(&artifact, artifact_root, "resume artifact").await;
    }
}

async fn remove_owned_file(
    path: &Path,
    owned_root: Option<&Path>,
    kind: &str,
) {
    let Some(owned_root) = owned_root else {
        tracing::warn!(path = %path.display(), "refusing to remove {kind} without an owned parent");
        return;
    };
    if let Err(error) = reject_symlink_components(owned_root).await {
        tracing::warn!(path = %path.display(), %error, "refusing to remove {kind} through a symlinked ancestor");
        return;
    }
    remove_file(path).await;
}

async fn remove_download_files(config: &DownloadConfig) -> Vec<DownloadCleanupFailure> {
    if let Err(failure) = validate_destructive_cleanup_paths(config).await {
        return vec![failure];
    }

    let paths = [
        config.resume_artifact_path("part"),
        config.resume_artifact_path("resume_data"),
        config.installation_artifact_path(),
        config.recovery_metadata_path(),
        config.recovery_metadata_staging_path(),
        config.destination.clone(),
        config.integrity_receipt_path(),
    ];
    let mut failures = Vec::new();
    for path in paths {
        if let Err(error) = remove_file_if_exists(&path).await {
            failures.push(DownloadCleanupFailure::new(&path, &error));
        }
    }
    failures
}

async fn validate_destructive_cleanup_paths(config: &DownloadConfig) -> Result<(), DownloadCleanupFailure> {
    let paths = [config.destination.parent().map(Path::to_path_buf), Some(config.artifact_root.clone())];
    for path in paths.into_iter().flatten() {
        if let Err(error) = reject_symlink_components(&path).await {
            let error = std::io::Error::other(error.to_string());
            return Err(DownloadCleanupFailure::new(&path, &error));
        }
    }
    Ok(())
}

async fn remove_directory_if_empty(path: &Path) -> Result<(), std::io::Error> {
    match tokio::fs::remove_dir(path).await {
        Ok(()) => Ok(()),
        Err(error) if matches!(error.kind(), ErrorKind::NotFound | ErrorKind::DirectoryNotEmpty) => Ok(()),
        Err(error) => Err(error),
    }
}

async fn remove_file_if_exists(path: &Path) -> Result<(), std::io::Error> {
    match fs::asyn::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[derive(Debug, thiserror::Error)]
enum CompletedFileValidationError {
    #[error("{0}")]
    InvalidContent(String),
    #[error(transparent)]
    Verification(#[from] VerificationError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl CompletedFileValidationError {
    fn destination_is_invalid(&self) -> bool {
        matches!(self, Self::InvalidContent(_))
    }

    fn into_download_error(
        self,
        path: &Path,
    ) -> DownloadError {
        match self {
            Self::InvalidContent(message) => DownloadError::IntegrityMismatch(message),
            Self::Verification(VerificationError::InvalidExpectedDigest {
                algorithm,
            }) => DownloadError::InvalidDigest {
                algorithm,
            },
            Self::Verification(VerificationError::Io(error)) | Self::Io(error) => DownloadError::IntegrityIo {
                path: path.display().to_string(),
                message: error.to_string(),
            },
        }
    }
}

async fn validate_completed_file(config: &DownloadConfig) -> Result<u64, CompletedFileValidationError> {
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
            Err(err) => return Err(err.into()),
        };
        if is_ready {
            break;
        }

        kiban::time::sleep(Duration::from_millis(50)).await;
    }

    let actual_bytes = fs::asyn::file_length(config.destination.as_path()).await?;
    if let Some(expected_bytes) = config.expected_bytes
        && expected_bytes != actual_bytes
    {
        return Err(CompletedFileValidationError::InvalidContent(format!(
            "downloaded file is {actual_bytes} bytes but registry declared {expected_bytes}"
        )));
    }

    let total_bytes = config.expected_bytes.unwrap_or(actual_bytes);

    match verify_file_integrity(&config.destination, &config.file_check).await? {
        VerificationStatus::Match => {
            if reject_symlink_components(&config.artifact_root).await.is_ok() {
                let _ =
                    save_integrity_cache_at(&config.destination, &config.file_check, &config.integrity_receipt_path())
                        .await;
            }
            Ok(total_bytes)
        },
        VerificationStatus::Mismatch => Err(CompletedFileValidationError::InvalidContent(
            config.file_check.verification_failure_message().to_string(),
        )),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/file_download_task_actor/download_task_actor_test.rs"]
mod tests;
