use std::path::PathBuf;

use tokio::{
    sync::{
        Mutex as TokioMutex,
        broadcast::Sender as TokioBroadcastSender,
        watch::{Receiver as TokioWatchReceiver, Sender as TokioWatchSender, channel as tokio_watch_channel},
    },
    time::{Duration as TokioDuration, sleep as tokio_sleep},
};
use uuid::Uuid;

use crate::{
    DownloadError, DownloadEventSender, DownloadId, DownloadInfo, FileCheck, FileDownloadEvent, FileDownloadPhase,
    FileDownloadState, FileDownloadTask as FileDownloadTaskTrait, InternalDownloadState, LockFileState,
    StateTransitionAction, check_lock_file,
    lock_manager::{acquire_lock, release_lock},
    managers::apple::{
        URLSessionDownloadTaskResumeData, URLSessionExt, UrlSessionDownloadTaskExt,
        url_session_state_reducer::{
            check_crc_file_exists, check_file_exists, check_resume_file_exists, reconcile_to_internal_state,
            reduce_to_checked_file_state, reduce_to_file_download_state,
        },
    },
    prelude::*,
};

pub struct FileDownloadTask {
    pub download_id: Uuid,
    pub source_url: String,
    pub destination: PathBuf,
    pub file_check: FileCheck,
    pub manager_id: String,
    expected_bytes: Option<u64>,
    state: Arc<TokioMutex<FileDownloadState>>,
    broadcast_sender: TokioBroadcastSender<FileDownloadState>,
    pub internal_state: TokioMutex<InternalDownloadState<Retained<NSURLSessionDownloadTask>>>,
    session: Option<Retained<NSURLSession>>,
    listener_task: Arc<TokioMutex<Option<TokioJoinHandle<()>>>>,
    tokio_handle: TokioHandle,
    completed_tx: TokioWatchSender<bool>,
    _completed_rx: TokioWatchReceiver<bool>,
}

impl std::fmt::Debug for FileDownloadTask {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("FileDownloadTask")
            .field("download_id", &self.download_id)
            .field("source_url", &self.source_url)
            .field("destination", &self.destination)
            .field("file_check", &self.file_check)
            .finish()
    }
}

impl FileDownloadTask {
    pub fn new(
        download_id: Uuid,
        source_url: String,
        destination: PathBuf,
        file_check: FileCheck,
        manager_id: String,
        expected_bytes: Option<u64>,
        internal_state: InternalDownloadState<Retained<NSURLSessionDownloadTask>>,
        initial_state: FileDownloadState,
        session: Option<Retained<NSURLSession>>,
        tokio_handle: TokioHandle,
    ) -> Self {
        let (broadcast_sender, _receiver) = tokio_broadcast_channel::<FileDownloadState>(64);
        let (completed_tx, completed_rx) = tokio_watch_channel(false);
        Self {
            download_id,
            source_url,
            destination,
            file_check,
            manager_id,
            expected_bytes,
            state: Arc::new(TokioMutex::new(initial_state)),
            broadcast_sender,
            internal_state: TokioMutex::new(internal_state),
            session,
            listener_task: Arc::new(TokioMutex::new(None)),
            tokio_handle,
            completed_tx,
            _completed_rx: completed_rx,
        }
    }

    pub async fn wait(&self) {
        let mut rx = self.completed_tx.subscribe();
        if *rx.borrow() {
            return;
        }
        let _ = rx.changed().await;
    }

    fn lock_path(&self) -> PathBuf {
        PathBuf::from(format!("{}.lock", self.destination.display()))
    }

    fn resume_data_path(&self) -> PathBuf {
        PathBuf::from(format!("{}.resume_data", self.destination.display()))
    }

    /// Reconcile lock state based on current internal state
    async fn reconcile_lock_state(&self) {
        let lock_path = self.lock_path();
        let lock_state = check_lock_file(&lock_path, &self.manager_id, std::process::id());

        let is_downloading = {
            let guard = self.internal_state.lock().await;
            matches!(*guard, InternalDownloadState::Downloading { .. })
        };

        if is_downloading {
            // Ensure we own the lock when downloading
            match lock_state {
                LockFileState::OwnedByUs(_) => {
                    // Already owned
                },
                LockFileState::Missing | LockFileState::OwnedBySameAppOldProcess(_) | LockFileState::Stale(_) => {
                    if let Err(e) = acquire_lock(&self.tokio_handle, &lock_path, &self.manager_id).await {
                        tracing::warn!(
                            "[FILE_TASK] Failed to acquire lock during reconciliation {}: {}",
                            lock_path.display(),
                            e
                        );
                    } else {
                        tracing::info!("[FILE_TASK] Acquired lock during reconciliation: {}", lock_path.display());
                    }
                },
                LockFileState::OwnedByOtherApp(info) => {
                    tracing::warn!("[FILE_TASK] Another manager owns the lock ({}).", info.manager_id);
                },
            }
        } else {
            // Not downloading → release our lock if we own it or it's safe to clean up
            match lock_state {
                LockFileState::OwnedByUs(_) | LockFileState::OwnedBySameAppOldProcess(_) | LockFileState::Stale(_) => {
                    if let Err(e) = release_lock(&self.tokio_handle, &lock_path).await {
                        tracing::warn!(
                            "[FILE_TASK] Failed to release lock during reconciliation {}: {}",
                            lock_path.display(),
                            e
                        );
                    } else {
                        tracing::info!("[FILE_TASK] Released lock during reconciliation: {}", lock_path.display());
                    }
                },
                LockFileState::Missing | LockFileState::OwnedByOtherApp(_) => {},
            }
        }
    }

    pub fn broadcast_sender(&self) -> TokioBroadcastSender<FileDownloadState> {
        self.broadcast_sender.clone()
    }

    pub async fn update_state_and_broadcast(
        &self,
        new_state: FileDownloadState,
    ) {
        let mut state_guard = self.state.lock().await;
        let current_bytes = state_guard.downloaded_bytes;
        let new_bytes = new_state.downloaded_bytes;

        tracing::debug!(
            "[FILE_TASK] update_state_and_broadcast: id={}, current={:?}({} bytes), new={:?}({} bytes)",
            self.download_id,
            state_guard.phase,
            current_bytes,
            new_state.phase,
            new_bytes
        );

        // Guard against backwards progress (stale NSURLSession events)
        // Only accept updates that maintain or increase progress
        if new_bytes < current_bytes
            && !matches!(
                new_state.phase,
                FileDownloadPhase::Downloaded | FileDownloadPhase::Error(_) | FileDownloadPhase::NotDownloaded
            )
        {
            tracing::debug!("[FILE_TASK] ⚠️ Rejected backwards progress: {} < {}", new_bytes, current_bytes);
            return;
        }

        // Allow state transitions to proceed normally
        // The guard against backwards progress above is sufficient

        *state_guard = new_state.clone();
        drop(state_guard);

        let receiver_count = self.broadcast_sender.receiver_count();

        if receiver_count == 0 {
            tracing::debug!("[FILE_TASK] No receivers active, skipping broadcast (likely shutdown)");
            return;
        }

        tracing::debug!("[FILE_TASK] ✓ State updated and broadcasting to {} receiver(s)", receiver_count);

        let _ = self.broadcast_sender.send(new_state.clone());
    }

    pub async fn download(&self) -> Result<(), DownloadError> {
        let mut internal_state_guard = self.internal_state.lock().await;
        internal_state_guard.can_transition(StateTransitionAction::Download)?;

        tracing::debug!(
            "[FILE_TASK] download() called: id={}, internal_state={:?}",
            self.download_id,
            match &*internal_state_guard {
                InternalDownloadState::NotDownloaded => "NotDownloaded",
                InternalDownloadState::Downloading {
                    ..
                } => "Downloading",
                InternalDownloadState::Paused {
                    ..
                } => "Paused",
                InternalDownloadState::Downloaded {
                    ..
                } => "Downloaded",
            }
        );

        match &*internal_state_guard {
            InternalDownloadState::NotDownloaded => {
                tracing::debug!("[FILE_TASK] Creating new download task for: {}", self.source_url);
                let session = self.session.as_ref().ok_or(DownloadError::TaskNotFoundAfterCreation)?;
                let download_task = session.download_task_with_url(&self.source_url)?;

                let download_info = match &self.file_check {
                    FileCheck::CRC(expected_crc) => DownloadInfo::with_crc(
                        self.source_url.clone(),
                        self.destination.display().to_string(),
                        expected_crc.clone(),
                    ),
                    FileCheck::None => {
                        DownloadInfo::new(self.source_url.clone(), self.destination.display().to_string())
                    },
                };
                download_task.set_download_info(&download_info);
                download_task.resume();

                *internal_state_guard = InternalDownloadState::Downloading {
                    task: download_task.clone(),
                };
                drop(internal_state_guard);

                // Ensure invariant: when downloading, no leftover resume_data exists
                let resume_data_path = self.resume_data_path();
                if resume_data_path.exists() {
                    let _ = fs::remove_file(&resume_data_path).await;
                    tracing::debug!("[FILE_TASK] Removed stale resume_data on start: {}", resume_data_path.display());
                }

                let downloading_state = FileDownloadState::downloading(0, 0);
                self.update_state_and_broadcast(downloading_state).await;
                // Ensure lock acquired now that we're downloading
                self.reconcile_lock_state().await;
                Ok(())
            },
            InternalDownloadState::Downloading {
                task: existing_download_task,
            } => {
                tracing::debug!("[FILE_TASK] Already downloading, calling resume on existing task");
                existing_download_task.resume();
                Ok(())
            },
            InternalDownloadState::Paused {
                part_path: saved_resume_path,
            } => {
                tracing::debug!("[FILE_TASK] Resuming from paused state, resume_path={:?}", saved_resume_path);

                let resume_path = saved_resume_path.clone();
                let total_bytes_fallback = self.expected_bytes.unwrap_or(0);

                async fn reset_to_not_downloaded_and_retry(
                    task: &FileDownloadTask,
                    resume_path: &PathBuf,
                    total_bytes: u64,
                    reason: &str,
                ) -> Result<(), DownloadError> {
                    tracing::warn!(
                        "[FILE_TASK] Resume data invalid ({}). Falling back to fresh download: {}",
                        reason,
                        resume_path.display()
                    );

                    // Best-effort cleanup: remove the unusable resume file so next launch doesn't
                    // keep presenting a Paused state that cannot actually be resumed.
                    let _ = fs::remove_file(resume_path).await;

                    // Reset user-facing state so progress can restart from 0 without being rejected
                    // by the backwards-progress guard.
                    task.update_state_and_broadcast(FileDownloadState::not_downloaded(total_bytes)).await;

                    // Ensure any lock is released before starting a fresh task.
                    task.reconcile_lock_state().await;

                    Box::pin(task.download()).await
                }

                // Check if resume file actually exists
                if !resume_path.exists() {
                    *internal_state_guard = InternalDownloadState::NotDownloaded;
                    drop(internal_state_guard);
                    return reset_to_not_downloaded_and_retry(
                        self,
                        &resume_path,
                        total_bytes_fallback,
                        "resume file missing",
                    )
                    .await;
                }

                // Log resume file size and destination state prior to resuming
                if let Ok(meta) = std::fs::metadata(&resume_path) {
                    tracing::debug!("[FILE_TASK] Resume data present: {} bytes", meta.len());
                }
                if let Ok(meta) = std::fs::metadata(&self.destination) {
                    tracing::debug!("[FILE_TASK] Destination prior to resume: exists with {} bytes", meta.len());
                } else {
                    tracing::debug!("[FILE_TASK] Destination prior to resume: missing");
                }

                let session = self.session.as_ref().ok_or(DownloadError::TaskNotFoundAfterCreation)?;
                let resume_data = match URLSessionDownloadTaskResumeData::from_file(&resume_path) {
                    Ok(data) => data,
                    Err(_e) => {
                        *internal_state_guard = InternalDownloadState::NotDownloaded;
                        drop(internal_state_guard);
                        return reset_to_not_downloaded_and_retry(
                            self,
                            &resume_path,
                            total_bytes_fallback,
                            "failed to parse resume file",
                        )
                        .await;
                    },
                };

                // If the resume blob points at a CFNetwork temp file that no longer exists
                // (very common after reboot), URLSession resume will fail or restart from 0.
                // Detect that early and restart cleanly to avoid a "stuck" Paused state.
                if let Some(temp_path) = resume_data.temp_file_path() {
                    if !temp_path.exists() {
                        tracing::warn!("[FILE_TASK] Resume temp file missing: {}", temp_path.display());
                        *internal_state_guard = InternalDownloadState::NotDownloaded;
                        drop(internal_state_guard);
                        return reset_to_not_downloaded_and_retry(
                            self,
                            &resume_path,
                            total_bytes_fallback,
                            "CFNetwork temp file missing",
                        )
                        .await;
                    }
                }

                let download_task = match session.download_task_with_resume_data(&resume_data) {
                    Ok(task) => task,
                    Err(_e) => {
                        *internal_state_guard = InternalDownloadState::NotDownloaded;
                        drop(internal_state_guard);
                        return reset_to_not_downloaded_and_retry(
                            self,
                            &resume_path,
                            total_bytes_fallback,
                            "failed to create URLSession task from resume data",
                        )
                        .await;
                    },
                };

                let download_info = match &self.file_check {
                    FileCheck::CRC(expected_crc) => DownloadInfo::with_crc(
                        self.source_url.clone(),
                        self.destination.display().to_string(),
                        expected_crc.clone(),
                    ),
                    FileCheck::None => {
                        DownloadInfo::new(self.source_url.clone(), self.destination.display().to_string())
                    },
                };
                download_task.set_download_info(&download_info);
                download_task.resume();

                let downloaded_bytes = resume_data.bytes_received.unwrap_or(0);
                let total_bytes = resume_data.bytes_expected_to_receive.unwrap_or(0);

                tracing::debug!(
                    "[FILE_TASK] Resuming with bytes_received={} of expected={}",
                    downloaded_bytes,
                    total_bytes
                );

                *internal_state_guard = InternalDownloadState::Downloading {
                    task: download_task.clone(),
                };
                drop(internal_state_guard);

                // Ensure invariant: when downloading, resume_data file is removed
                let resume_data_path = self.resume_data_path();
                if resume_data_path.exists() {
                    let _ = fs::remove_file(&resume_data_path).await;
                    tracing::debug!("[FILE_TASK] Removed resume_data after resume: {}", resume_data_path.display());
                }

                let downloading_state = FileDownloadState::downloading(downloaded_bytes, total_bytes);
                self.update_state_and_broadcast(downloading_state).await;
                // Ensure lock acquired now that we're downloading
                self.reconcile_lock_state().await;
                Ok(())
            },
            InternalDownloadState::Downloaded {
                ..
            } => {
                tracing::debug!("[FILE_TASK] Already downloaded, broadcasting current state");
                let current_state = self.state.lock().await.clone();
                self.update_state_and_broadcast(current_state).await;
                let _ = self.completed_tx.send(true);
                Ok(())
            },
        }
    }

    pub async fn pause(&self) -> Result<(), DownloadError> {
        let mut internal_state_guard = self.internal_state.lock().await;
        internal_state_guard.can_transition(StateTransitionAction::Pause)?;

        tracing::debug!(
            "[FILE_TASK] pause() called: id={}, internal_state={:?}",
            self.download_id,
            match &*internal_state_guard {
                InternalDownloadState::NotDownloaded => "NotDownloaded",
                InternalDownloadState::Downloading {
                    ..
                } => "Downloading",
                InternalDownloadState::Paused {
                    ..
                } => "Paused",
                InternalDownloadState::Downloaded {
                    ..
                } => "Downloaded",
            }
        );

        match &*internal_state_guard {
            InternalDownloadState::Downloading {
                task: download_task,
            } => {
                tracing::debug!("[FILE_TASK] Pausing download task, producing resume data...");

                let resume_data = download_task.cancel_by_producing_resume_data().await?;
                let resume_data_path = format!("{}.resume_data", self.destination.display());
                resume_data.save_to_file(&resume_data_path).map_err(|_| DownloadError::ResumeDataError)?;

                let downloaded_bytes = resume_data.bytes_received.unwrap_or(0);
                let total_bytes = resume_data.bytes_expected_to_receive.unwrap_or(0);

                tracing::debug!("[FILE_TASK] Resume data saved: {} bytes, transitioning to Paused", downloaded_bytes);

                *internal_state_guard = InternalDownloadState::Paused {
                    part_path: resume_data_path.into(),
                };
                drop(internal_state_guard);

                let paused_state = FileDownloadState::paused(downloaded_bytes, total_bytes);
                self.update_state_and_broadcast(paused_state).await;
                // Ensure lock released once paused
                self.reconcile_lock_state().await;

                Ok(())
            },
            InternalDownloadState::Downloaded {
                ..
            } => Ok(()),
            _ => Ok(()),
        }
    }

    pub async fn cancel(&self) -> Result<(), DownloadError> {
        let mut internal_state_guard = self.internal_state.lock().await;
        internal_state_guard.can_transition(StateTransitionAction::Cancel)?;

        tracing::debug!(
            "[FILE_TASK] cancel() called: id={}, internal_state before={:?}",
            self.download_id,
            match &*internal_state_guard {
                InternalDownloadState::NotDownloaded => "NotDownloaded",
                InternalDownloadState::Downloading {
                    ..
                } => "Downloading",
                InternalDownloadState::Paused {
                    ..
                } => "Paused",
                InternalDownloadState::Downloaded {
                    ..
                } => "Downloaded",
            }
        );

        let resume_path_to_delete: Option<PathBuf> = match &*internal_state_guard {
            InternalDownloadState::Downloading {
                task: download_task,
            } => {
                download_task.cancel();
                Some(PathBuf::from(format!("{}.resume_data", self.destination.display())))
            },
            InternalDownloadState::Paused {
                part_path: saved_resume_path,
            } => Some(saved_resume_path.clone()),
            _ => None,
        };
        *internal_state_guard = InternalDownloadState::NotDownloaded;
        drop(internal_state_guard);

        if let Some(resume_path) = resume_path_to_delete {
            let _ = fs::remove_file(&resume_path).await;
        }

        tracing::debug!("[FILE_TASK] cancel() completed: id={}, internal_state now=NotDownloaded", self.download_id);

        let not_downloaded_state = FileDownloadState::not_downloaded(0);
        self.update_state_and_broadcast(not_downloaded_state).await;
        // Ensure lock released once canceled
        self.reconcile_lock_state().await;
        Ok(())
    }

    pub async fn state(&self) -> FileDownloadState {
        let state = self.state.lock().await.clone();
        state
    }

    pub async fn progress(&self) -> Result<TokioBroadcastStream<FileDownloadState>, DownloadError> {
        Ok(TokioBroadcastStream::new(self.broadcast_sender.subscribe()))
    }

    /// Handle URLSession download completion event
    /// This encapsulates the reduction and reconciliation logic
    pub async fn handle_download_completion(&self) {
        tracing::debug!(
            "[FILE_TASK] handle_download_completion: id={}, dest={}",
            self.download_id,
            self.destination.display()
        );

        // Small delay to ensure filesystem operations complete
        // macOS file system might have slight latency
        self.tokio_handle
            .spawn(async {
                tokio_sleep(TokioDuration::from_millis(10)).await;
            })
            .await
            .ok();

        let crc_file_path = PathBuf::from(format!("{}.crc", self.destination.display()));
        let resume_data_path = PathBuf::from(format!("{}.resume_data", self.destination.display()));

        // Check file states
        let downloaded_file_state = check_file_exists(&self.destination);
        let crc_file_state = check_crc_file_exists(&crc_file_path);
        let resume_data_state = check_resume_file_exists(&resume_data_path);

        tracing::debug!(
            "[FILE_TASK] File states: downloaded={:?}, crc={:?}, resume={:?}",
            downloaded_file_state,
            crc_file_state,
            resume_data_state
        );
        tracing::debug!("[FILE_TASK] Destination file exists (explicit check): {}", self.destination.exists());
        if let Ok(metadata) = std::fs::metadata(&self.destination) {
            tracing::debug!("[FILE_TASK] Destination file size: {} bytes", metadata.len());
        }

        let expected_crc = match &self.file_check {
            FileCheck::CRC(crc_hash) => Some(crc_hash.as_str()),
            FileCheck::None => None,
        };

        // Reduce to checked file state (validates CRC if needed)
        let checked_file_state =
            reduce_to_checked_file_state(downloaded_file_state, crc_file_state, &self.destination, expected_crc);

        tracing::debug!("[FILE_TASK] Checked file state: {:?}", checked_file_state);

        // Reconcile to internal state (cleanup, etc.)
        let internal_state = reconcile_to_internal_state(
            checked_file_state,
            resume_data_state,
            None,
            &self.destination,
            &crc_file_path,
            &resume_data_path,
            self.expected_bytes,
        )
        .await;

        // Reduce to file download state for display
        let file_download_state = reduce_to_file_download_state(
            checked_file_state,
            resume_data_state,
            None,
            None,
            &self.destination,
            self.expected_bytes,
            &self.manager_id,
        );

        tracing::debug!("[FILE_TASK] Final state after completion: {:?}", file_download_state.phase);

        // Update internal state
        let mut internal_state_guard = self.internal_state.lock().await;
        *internal_state_guard = internal_state;
        drop(internal_state_guard);

        // Broadcast the new state
        self.update_state_and_broadcast(file_download_state).await;

        // Reconcile lock after completion (should release)
        self.reconcile_lock_state().await;
    }

    /// Handle URLSession progress update event
    pub async fn handle_progress_update(
        &self,
        _bytes_written: u64,
        total_bytes_written: u64,
        total_bytes_expected: u64,
    ) {
        tracing::debug!(
            "[FILE_TASK] handle_progress_update: id={}, progress={}/{} bytes",
            self.download_id,
            total_bytes_written,
            total_bytes_expected
        );

        let resume_data_path = self.resume_data_path();
        if resume_data_path.exists() {
            if let Err(e) = fs::remove_file(&resume_data_path).await {
                tracing::warn!(
                    "[FILE_TASK] Failed to remove stale resume_data on progress {}: {}",
                    resume_data_path.display(),
                    e
                );
            } else {
                tracing::debug!("[FILE_TASK] Removed stale resume_data on progress: {}", resume_data_path.display());
            }
        }

        let downloading_state = FileDownloadState::downloading(total_bytes_written, total_bytes_expected);
        self.update_state_and_broadcast(downloading_state).await;
    }

    /// Handle URLSession error event
    pub async fn handle_error(
        &self,
        error_message: String,
    ) {
        let error_state = FileDownloadState::error(error_message);
        self.update_state_and_broadcast(error_state).await;
        // On errors, ensure lock is released
        self.reconcile_lock_state().await;
    }

    /// Start listening to global download events
    pub async fn start_listening(
        &self,
        global_broadcast: DownloadEventSender,
    ) {
        // Check if already listening
        let mut listener_guard = self.listener_task.lock().await;
        if listener_guard.is_some() {
            tracing::debug!("[FILE_TASK] start_listening: Already listening for id={}", self.download_id);
            return;
        }

        tracing::debug!("[FILE_TASK] start_listening: Starting listener for id={}", self.download_id);

        let download_id = self.download_id;
        let task = Arc::new(self.clone_arc_fields());
        let completed = self.completed_tx.clone();

        tracing::debug!("[FILE_TASK] Spawning listener task for download_id={}", download_id);

        let handle = self.tokio_handle.spawn(async move {
            tracing::debug!("[FILE_TASK] Listener task started for id={}", download_id);

            let mut stream = TokioBroadcastStream::new(global_broadcast.subscribe());

            while let Some(result) = stream.next().await {
                if let Ok((id, event)) = result {
                    if id == download_id {
                        tracing::debug!("[FILE_TASK] Received event for download_id={}: {:?}", download_id, event);

                        match event {
                            FileDownloadEvent::DownloadCompleted {
                                final_destination,
                                ..
                            } => {
                                tracing::debug!(
                                    "[FILE_TASK] Processing DownloadCompleted: id={}, reported_dest={}",
                                    download_id,
                                    final_destination.display()
                                );

                                tracing::debug!("[FILE_TASK] Task's destination path: {}", task.destination.display());

                                tracing::debug!("[FILE_TASK] Paths match: {}", final_destination == task.destination);

                                // Add a small delay to ensure file system operations are complete
                                // The delegate moves files synchronously, but filesystem flush might be async
                                let _ = task
                                    .tokio_handle
                                    .spawn(async {
                                        tokio_sleep(TokioDuration::from_millis(100)).await;
                                    })
                                    .await;

                                task.handle_download_completion().await;
                                let _ = completed.send(true);
                            },
                            FileDownloadEvent::ProgressUpdate {
                                bytes_written,
                                total_bytes_written,
                                total_bytes_expected,
                            } => {
                                task.handle_progress_update(bytes_written, total_bytes_written, total_bytes_expected)
                                    .await;
                            },
                            FileDownloadEvent::Error {
                                message,
                            } => {
                                task.handle_error(message).await;
                                let _ = completed.send(true);
                            },
                        }
                    }
                }
            }
        });

        *listener_guard = Some(handle);
    }

    /// Stop listening to global download events
    pub async fn stop_listening(&self) {
        let mut listener_guard = self.listener_task.lock().await;
        if let Some(handle) = listener_guard.take() {
            handle.abort();
            let _ = handle.await;
        }
    }

    /// Helper to clone Arc fields for listener task
    fn clone_arc_fields(&self) -> Self {
        Self {
            download_id: self.download_id,
            source_url: self.source_url.clone(),
            destination: self.destination.clone(),
            file_check: self.file_check.clone(),
            manager_id: self.manager_id.clone(),
            expected_bytes: self.expected_bytes,
            state: self.state.clone(),
            broadcast_sender: self.broadcast_sender.clone(),
            internal_state: TokioMutex::new(InternalDownloadState::NotDownloaded),
            session: self.session.clone(),
            listener_task: Arc::new(TokioMutex::new(None)),
            tokio_handle: self.tokio_handle.clone(),
            completed_tx: self.completed_tx.clone(),
            _completed_rx: self.completed_tx.subscribe(),
        }
    }
}

#[async_trait::async_trait]
impl FileDownloadTaskTrait for FileDownloadTask {
    fn download_id(&self) -> DownloadId {
        self.download_id
    }

    fn source_url(&self) -> &str {
        &self.source_url
    }

    fn destination(&self) -> &std::path::Path {
        &self.destination
    }

    fn file_check(&self) -> &FileCheck {
        &self.file_check
    }

    fn broadcast_sender(&self) -> TokioBroadcastSender<FileDownloadState> {
        self.broadcast_sender()
    }

    async fn download(&self) -> Result<(), DownloadError> {
        self.download().await
    }

    async fn pause(&self) -> Result<(), DownloadError> {
        self.pause().await
    }

    async fn cancel(&self) -> Result<(), DownloadError> {
        self.cancel().await
    }

    async fn state(&self) -> FileDownloadState {
        self.state().await
    }

    async fn progress(&self) -> Result<TokioBroadcastStream<FileDownloadState>, DownloadError> {
        self.progress().await
    }

    async fn start_listening(
        &self,
        global_broadcast: DownloadEventSender,
    ) {
        self.start_listening(global_broadcast).await
    }

    async fn stop_listening(&self) {
        self.stop_listening().await
    }

    async fn wait(&self) {
        self.wait().await
    }
}
