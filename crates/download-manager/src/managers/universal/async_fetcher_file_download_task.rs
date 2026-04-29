use async_fetcher::{FetchEvent, Fetcher, Source};
use async_shutdown::ShutdownManager;
use tokio::{
    sync::{
        mpsc,
        watch::{Receiver as TokioWatchReceiver, Sender as TokioWatchSender, channel as tokio_watch_channel},
    },
    time::{Duration as TokioDuration, sleep as tokio_sleep},
};
use tokio_stream::{
    StreamExt as TokioStreamExt,
    wrappers::{BroadcastStream as TokioBroadcastStream, UnboundedReceiverStream},
};

use crate::{
    Arc, DownloadError, DownloadEventSender, DownloadId, FileCheck, FileDownloadEvent, FileDownloadPhase,
    FileDownloadState, FileDownloadTask as FileDownloadTaskTrait, InternalDownloadState, LockFileState, PathBuf,
    StateTransitionAction, TokioBroadcastSender, TokioHandle, TokioJoinHandle, TokioMutex, Uuid, acquire_lock,
    calculate_and_verify_crc, check_lock_file, crc_utils, fs, managers::universal::AsyncFetcherConfig, release_lock,
    tokio_broadcast_channel,
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
    pub internal_state: TokioMutex<InternalDownloadState<()>>,
    config: AsyncFetcherConfig,
    shutdown_manager: Arc<TokioMutex<Option<ShutdownManager<()>>>>,
    fetcher_task: Arc<TokioMutex<Option<TokioJoinHandle<()>>>>,
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
        internal_state: InternalDownloadState<()>,
        initial_state: FileDownloadState,
        config: AsyncFetcherConfig,
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
            config,
            shutdown_manager: Arc::new(TokioMutex::new(None)),
            fetcher_task: Arc::new(TokioMutex::new(None)),
            listener_task: Arc::new(TokioMutex::new(None)),
            tokio_handle,
            completed_tx,
            _completed_rx: completed_rx,
        }
    }

    fn lock_path(&self) -> PathBuf {
        PathBuf::from(format!("{}.lock", self.destination.display()))
    }

    async fn reconcile_lock_state(&self) {
        let lock_path = self.lock_path();
        let lock_state = check_lock_file(&lock_path, &self.manager_id, std::process::id());

        let is_downloading = {
            let guard = self.internal_state.lock().await;
            matches!(*guard, InternalDownloadState::Downloading { .. })
        };

        if is_downloading {
            match lock_state {
                LockFileState::OwnedByUs(_) => {},
                LockFileState::Missing | LockFileState::OwnedBySameAppOldProcess(_) | LockFileState::Stale(_) => {
                    if let Err(e) = acquire_lock(&self.tokio_handle, &lock_path, &self.manager_id).await {
                        tracing::warn!(
                            "[FILE_TASK] Failed to acquire lock during reconciliation {}: {}",
                            lock_path.display(),
                            e
                        );
                    }
                },
                LockFileState::OwnedByOtherApp(info) => {
                    tracing::warn!("[FILE_TASK] Another manager owns the lock ({}).", info.manager_id);
                },
            }
        } else {
            match lock_state {
                LockFileState::OwnedByUs(_) | LockFileState::OwnedBySameAppOldProcess(_) | LockFileState::Stale(_) => {
                    let _ = release_lock(&self.tokio_handle, &lock_path).await;
                },
                _ => {},
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

        if new_bytes < current_bytes
            && !matches!(
                new_state.phase,
                FileDownloadPhase::Downloaded | FileDownloadPhase::Error(_) | FileDownloadPhase::NotDownloaded
            )
        {
            tracing::debug!("[FILE_TASK] ⚠️ Rejected backwards progress: {} < {}", new_bytes, current_bytes);
            return;
        }

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
            InternalDownloadState::NotDownloaded
            | InternalDownloadState::Paused {
                ..
            } => {
                let part_path = self.destination.with_extension("part");

                if let Some(parent) = self.destination.parent() {
                    tokio::fs::create_dir_all(parent).await.map_err(|e| DownloadError::IOError(e.to_string()))?;
                }

                let resume_from_bytes = if part_path.exists() {
                    tokio::fs::metadata(&part_path).await.ok().map(|m| m.len()).unwrap_or(0)
                } else {
                    0
                };

                let shutdown = ShutdownManager::new();
                *self.shutdown_manager.lock().await = Some(shutdown.clone());

                let (events_tx, events_rx) = mpsc::unbounded_channel();

                let fetcher = Fetcher::<()>::default()
                    .connections_per_file(self.config.connections_per_file)
                    .retries(self.config.retries)
                    .progress_interval(self.config.progress_interval_ms)
                    .shutdown(shutdown.clone())
                    .events(events_tx)
                    .build();

                let source = Source::builder(Arc::from(self.destination.clone()), self.source_url.clone().into())
                    .partial(Arc::from(part_path.clone()))
                    .build();

                let download_id = self.download_id;
                let destination = self.destination.clone();
                let expected_crc = match &self.file_check {
                    FileCheck::CRC(crc) => Some(crc.clone()),
                    FileCheck::None => None,
                };
                let expected_bytes = self.expected_bytes;
                let broadcast_sender = self.broadcast_sender.clone();

                *internal_state_guard = InternalDownloadState::Downloading {
                    task: (),
                };
                drop(internal_state_guard);

                let downloading_state = FileDownloadState::downloading(resume_from_bytes, 0);
                self.update_state_and_broadcast(downloading_state).await;
                // Ensure lock acquired once downloading
                self.reconcile_lock_state().await;

                let _task = self.tokio_handle.spawn(async move {
                    tracing::info!("[FETCHER] Event processing task started for id={}", download_id);
                    let mut total_size: Option<u64> = None;
                    let mut downloaded_bytes = resume_from_bytes;

                    let mut evt_rx = UnboundedReceiverStream::new(events_rx);
                    while let Some((_path, _data, evt)) = evt_rx.next().await {
                        tracing::debug!(
                            "[FETCHER] Received event: id={}, event={:?}",
                            download_id,
                            std::mem::discriminant(&evt)
                        );
                        match evt {
                            FetchEvent::Fetching => {
                                tracing::debug!("[FETCHER] Started fetching: {}", download_id);
                            },
                            FetchEvent::ContentLength(len) => {
                                // Content-Length is the remaining bytes to download
                                // Add resume_from_bytes to get total file size
                                total_size = Some(len + resume_from_bytes);
                                tracing::debug!(
                                    "[FETCHER] Content length: {} bytes (remaining: {}, resumed: {})",
                                    len + resume_from_bytes,
                                    len,
                                    resume_from_bytes
                                );
                            },
                            FetchEvent::Progress(bytes) => {
                                // Progress reports incremental bytes since last update
                                // We need to accumulate them to get total downloaded
                                downloaded_bytes += bytes;

                                let total = total_size.unwrap_or(expected_bytes.unwrap_or(downloaded_bytes));
                                let progress_state = FileDownloadState::downloading(downloaded_bytes, total);
                                let receiver_count = broadcast_sender.receiver_count();
                                tracing::info!(
                                    "[FETCHER] Progress: id={}, bytes={}/{}, receivers={}",
                                    download_id,
                                    downloaded_bytes,
                                    total,
                                    receiver_count
                                );
                                let _ = broadcast_sender.send(progress_state);
                            },
                            FetchEvent::Fetched => {
                                tracing::debug!("[FETCHER] Download completed: {}", download_id);

                                // async-fetcher automatically moves the .part file to destination
                                // We just need to wait a moment for the file system to catch up
                                if !destination.exists() {
                                    tracing::warn!(
                                        "[FETCHER] Destination file not yet visible after Fetched event: {}",
                                        destination.display()
                                    );
                                    // Give a brief moment for filesystem operations to complete
                                    tokio_sleep(TokioDuration::from_millis(100)).await;

                                    if !destination.exists() {
                                        tracing::error!(
                                            "[FETCHER] Destination file still missing after wait: {}",
                                            destination.display()
                                        );
                                        let error_state = FileDownloadState {
                                            phase: FileDownloadPhase::Error(
                                                "Download completed but file not found at destination".to_string(),
                                            ),
                                            downloaded_bytes: 0,
                                            total_bytes: 0,
                                        };
                                        let _ = broadcast_sender.send(error_state);
                                        return;
                                    }
                                }

                                if let Some(ref crc) = expected_crc {
                                    tracing::debug!(
                                        "[FETCHER] Verifying CRC for {}, expected: {}",
                                        destination.display(),
                                        crc
                                    );

                                    if let Ok(metadata) = std::fs::metadata(&destination) {
                                        tracing::debug!("[FETCHER] File size: {} bytes", metadata.len());
                                    }

                                    match calculate_and_verify_crc(&destination, crc) {
                                        Ok(true) => {
                                            tracing::debug!("[FETCHER] CRC verification passed");

                                            // Save CRC file for future verification
                                            if let Err(e) = crc_utils::save_crc_file(&destination, crc) {
                                                tracing::warn!("[FETCHER] Failed to save CRC file: {}", e);
                                            }

                                            let final_bytes = downloaded_bytes;
                                            let total = total_size.unwrap_or(expected_bytes.unwrap_or(final_bytes));
                                            let completed_state = FileDownloadState {
                                                phase: FileDownloadPhase::Downloaded,
                                                downloaded_bytes: final_bytes,
                                                total_bytes: total,
                                            };
                                            let _ = broadcast_sender.send(completed_state);
                                        },
                                        Ok(false) => {
                                            tracing::error!("[FETCHER] CRC verification failed - checksum mismatch");
                                            let _ = fs::remove_file(&destination);
                                            let error_state = FileDownloadState {
                                                phase: FileDownloadPhase::Error("CRC verification failed".to_string()),
                                                downloaded_bytes: 0,
                                                total_bytes: 0,
                                            };
                                            let _ = broadcast_sender.send(error_state);
                                        },
                                        Err(e) => {
                                            tracing::error!("[FETCHER] CRC verification error: {}", e);
                                            let _ = fs::remove_file(&destination);
                                            let error_state = FileDownloadState {
                                                phase: FileDownloadPhase::Error(format!(
                                                    "CRC verification error: {}",
                                                    e
                                                )),
                                                downloaded_bytes: 0,
                                                total_bytes: 0,
                                            };
                                            let _ = broadcast_sender.send(error_state);
                                        },
                                    }
                                } else {
                                    let final_bytes = downloaded_bytes;
                                    let total = total_size.unwrap_or(expected_bytes.unwrap_or(final_bytes));
                                    let completed_state = FileDownloadState {
                                        phase: FileDownloadPhase::Downloaded,
                                        downloaded_bytes: final_bytes,
                                        total_bytes: total,
                                    };
                                    let _ = broadcast_sender.send(completed_state);
                                }
                            },
                            FetchEvent::Retrying => {
                                tracing::debug!("[FETCHER] Retrying download: {}", download_id);
                            },
                        }
                    }
                });

                let source_stream = futures_util::stream::once(async { (source, Arc::new(())) });
                let fetch_task = fetcher.stream_from(source_stream, 1);
                let broadcast_clone = self.broadcast_sender.clone();
                let download_id_clone = self.download_id;

                tracing::info!("[FETCHER] Starting fetch task for id={}", download_id);

                self.tokio_handle.spawn(async move {
                    futures_util::pin_mut!(fetch_task);
                    while let Some((_path, _data, result)) = fetch_task.next().await {
                        if let Err(e) = result {
                            tracing::error!("[FETCHER] Error downloading {}: {}", download_id_clone, e);
                            let error_state = FileDownloadState {
                                phase: FileDownloadPhase::Error(e.to_string()),
                                downloaded_bytes: 0,
                                total_bytes: 0,
                            };
                            let _ = broadcast_clone.send(error_state);
                        }
                    }
                });

                Ok(())
            },
            InternalDownloadState::Downloading {
                ..
            } => {
                tracing::debug!("[FILE_TASK] Already downloading");
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
                ..
            } => {
                tracing::debug!("[FILE_TASK] Triggering shutdown for pause...");

                if let Some(shutdown) = self.shutdown_manager.lock().await.as_ref() {
                    let _ = shutdown.trigger_shutdown(());
                }

                let part_path = self.destination.with_extension("part");
                let downloaded_bytes = tokio::fs::metadata(&part_path).await.ok().map(|m| m.len()).unwrap_or(0);
                let total_bytes = self.expected_bytes.unwrap_or(downloaded_bytes);

                *internal_state_guard = InternalDownloadState::Paused {
                    part_path: part_path.clone(),
                };
                drop(internal_state_guard);

                let paused_state = FileDownloadState::paused(downloaded_bytes, total_bytes);
                self.update_state_and_broadcast(paused_state).await;

                if let Some(task) = self.fetcher_task.lock().await.take() {
                    task.abort();
                    let _ = task.await;
                }
                // Reconcile lock (release)
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

        match &*internal_state_guard {
            InternalDownloadState::Downloading {
                ..
            } => {
                if let Some(shutdown) = self.shutdown_manager.lock().await.as_ref() {
                    let _ = shutdown.trigger_shutdown(());
                }

                if let Some(task) = self.fetcher_task.lock().await.take() {
                    task.abort();
                    let _ = task.await;
                }
            },
            InternalDownloadState::Paused {
                ..
            } => {},
            _ => {},
        }

        let part_path = self.destination.with_extension("part");
        let _ = fs::remove_file(&part_path);

        *internal_state_guard = InternalDownloadState::NotDownloaded;
        drop(internal_state_guard);

        tracing::debug!("[FILE_TASK] cancel() completed: id={}, internal_state now=NotDownloaded", self.download_id);

        let not_downloaded_state = FileDownloadState::not_downloaded(0);
        self.update_state_and_broadcast(not_downloaded_state).await;
        // Ensure lock released on cancel
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

    pub async fn wait(&self) {
        let mut rx = self.completed_tx.subscribe();
        if *rx.borrow() {
            return;
        }
        let _ = rx.changed().await;
    }

    pub async fn start_listening(
        &self,
        global_broadcast: DownloadEventSender,
    ) {
        let mut listener_guard = self.listener_task.lock().await;
        if listener_guard.is_some() {
            tracing::debug!("[FILE_TASK] start_listening: Already listening for id={}", self.download_id);
            return;
        }

        tracing::debug!("[FILE_TASK] start_listening: Starting listener for id={}", self.download_id);

        let download_id = self.download_id;
        let destination = self.destination.clone();
        let mut local_stream = TokioBroadcastStream::new(self.broadcast_sender.subscribe());
        let tokio_handle = self.tokio_handle.clone();
        let tokio_handle_for_closure = tokio_handle.clone();

        let manager_id = self.manager_id.clone();
        let completed = self.completed_tx.clone();
        let handle = tokio_handle.spawn(async move {
            tracing::debug!("[FILE_TASK] Listener task started for id={}", download_id);

            let mut last_downloaded_bytes = 0u64;

            while let Some(result) = tokio_stream::StreamExt::next(&mut local_stream).await {
                if let Ok(state) = result {
                    tracing::debug!("[FILE_TASK] Forwarding event for download_id={}: {:?}", download_id, state.phase);

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
                            // Reconcile lock after completion
                            let lock_path = PathBuf::from(format!("{}.lock", destination.display()));
                            let _ = tokio_handle_for_closure
                                .clone()
                                .spawn(async move {
                                    let lock_state = check_lock_file(&lock_path, &manager_id, std::process::id());
                                    match lock_state {
                                        LockFileState::OwnedByUs(_)
                                        | LockFileState::OwnedBySameAppOldProcess(_)
                                        | LockFileState::Stale(_) => {
                                            let _ = release_lock(&tokio_handle_for_closure, &lock_path).await;
                                        },
                                        _ => {},
                                    }
                                })
                                .await;

                            let event = FileDownloadEvent::DownloadCompleted {
                                tmp_path: destination.clone(),
                                final_destination: destination.clone(),
                            };
                            let _ = global_broadcast.send((download_id, event));
                            tracing::debug!("[FILE_TASK] Sent DownloadCompleted event for id={}", download_id);

                            let _ = completed.send(true);
                            break;
                        },
                        FileDownloadPhase::Error(msg) => {
                            // Reconcile lock on error
                            let lock_path = PathBuf::from(format!("{}.lock", destination.display()));
                            let _ = tokio_handle_for_closure
                                .clone()
                                .spawn(async move {
                                    let lock_state = check_lock_file(&lock_path, &manager_id, std::process::id());
                                    match lock_state {
                                        LockFileState::OwnedByUs(_)
                                        | LockFileState::OwnedBySameAppOldProcess(_)
                                        | LockFileState::Stale(_) => {
                                            let _ = release_lock(&tokio_handle_for_closure, &lock_path).await;
                                        },
                                        _ => {},
                                    }
                                })
                                .await;

                            let event = FileDownloadEvent::Error {
                                message: msg,
                            };
                            let _ = global_broadcast.send((download_id, event));

                            let _ = completed.send(true);
                            break;
                        },
                        _ => {},
                    }
                }
            }

            tracing::debug!("[FILE_TASK] Listener task ended for id={}", download_id);
        });

        *listener_guard = Some(handle);
    }

    pub async fn stop_listening(&self) {
        let mut listener_guard = self.listener_task.lock().await;
        if let Some(handle) = listener_guard.take() {
            handle.abort();
            let _ = handle.await;
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
