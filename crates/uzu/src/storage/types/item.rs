use std::{
    fs::{create_dir_all, read_dir, remove_dir_all, remove_file},
    path::{Path, PathBuf},
    sync::Arc,
};

use download_manager::{FileDownloadManager, FileDownloadState, FileDownloadTask};
use futures_util::future::join_all;
use shoji::types::basic::File;
use tokio::{
    runtime::Handle,
    sync::broadcast::{Sender, channel},
    task::JoinHandle,
};
use tokio_stream::wrappers::BroadcastStream;

use crate::{
    helpers::SharedAccess,
    storage::{
        Error,
        types::{CrcSnapshot, DownloadPhase, DownloadState, reduce_file_download_states},
    },
};

pub struct Item {
    pub identifier: String,
    pub files: Arc<Vec<File>>,
    pub cache_path: PathBuf,

    download_state: SharedAccess<DownloadState>,
    file_download_manager: Arc<dyn FileDownloadManager>,
    file_download_tasks: SharedAccess<Vec<Arc<dyn FileDownloadTask>>>,
    file_download_states: SharedAccess<Vec<FileDownloadState>>,

    handle: Handle,
    broadcast_sender: Sender<DownloadState>,
    storage_broadcast_sender: Sender<(String, DownloadState)>,
    listener_task: SharedAccess<Option<JoinHandle<()>>>,
}

impl std::fmt::Debug for Item {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("Item").field("identifier", &self.identifier).field("cache_path", &self.cache_path).finish()
    }
}

impl Clone for Item {
    fn clone(&self) -> Self {
        Self {
            identifier: self.identifier.clone(),
            files: self.files.clone(),
            cache_path: self.cache_path.clone(),
            download_state: self.download_state.clone(),
            file_download_manager: self.file_download_manager.clone(),
            file_download_tasks: self.file_download_tasks.clone(),
            file_download_states: self.file_download_states.clone(),
            handle: self.handle.clone(),
            broadcast_sender: self.broadcast_sender.clone(),
            storage_broadcast_sender: self.storage_broadcast_sender.clone(),
            listener_task: self.listener_task.clone(),
        }
    }
}

impl Item {
    pub fn new(
        identifier: String,
        files: Arc<Vec<File>>,
        cache_path: PathBuf,
        download_state: DownloadState,
        file_download_manager: Arc<dyn FileDownloadManager>,
        file_download_tasks: Vec<Arc<dyn FileDownloadTask>>,
        handle: Handle,
        storage_broadcast_sender: Sender<(String, DownloadState)>,
    ) -> Self {
        let (broadcast_sender, _) = channel(64);
        let file_download_states = SharedAccess::new(Vec::new());
        Self {
            identifier,
            files,
            cache_path,
            download_state: SharedAccess::new(download_state),
            file_download_manager,
            file_download_tasks: SharedAccess::new(file_download_tasks),
            file_download_states,
            handle,
            broadcast_sender,
            storage_broadcast_sender,
            listener_task: SharedAccess::new(None),
        }
    }

    pub async fn state(&self) -> DownloadState {
        let state = self.download_state.lock().await.clone();
        state
    }

    async fn get_file_download_states(&self) -> Vec<FileDownloadState> {
        let file_tasks_guard = self.file_download_tasks.lock().await;
        let num_file_tasks = file_tasks_guard.len();
        drop(file_tasks_guard);

        // Use cached states from broadcasts - more reliable than querying
        let cache_guard = self.file_download_states.lock().await;

        // Validate cache is properly sized before using it
        if !cache_guard.is_empty() && cache_guard.len() == num_file_tasks {
            let mut states = cache_guard.clone();
            for (i, state) in states.iter_mut().enumerate() {
                if state.total_bytes == 0 {
                    if let Some(file_info) = self.files.get(i) {
                        if file_info.size > 0 {
                            state.total_bytes = file_info.size as u64;
                        }
                    }
                }
            }
            return states;
        }

        // If cache exists but size mismatched, fall back to direct query
        if !cache_guard.is_empty() && cache_guard.len() != num_file_tasks {
            tracing::warn!(
                "[MODEL] Cache size mismatch: cache={}, tasks={}, model={}. Falling back to direct query.",
                cache_guard.len(),
                num_file_tasks,
                self.identifier
            );
        }

        drop(cache_guard);

        // Fall back to querying if cache not initialized or size mismatch
        let file_tasks_guard = self.file_download_tasks.lock().await;
        let mut states = Vec::new();
        for file_task in file_tasks_guard.iter() {
            states.push(file_task.state().await);
        }

        // Patch total_bytes from registry if missing (safety net)
        for (i, state) in states.iter_mut().enumerate() {
            if state.total_bytes == 0 {
                if let Some(file_info) = self.files.get(i) {
                    if file_info.size > 0 {
                        state.total_bytes = file_info.size as u64;
                    }
                }
            }
        }

        states
    }

    pub async fn reduce_state(&self) -> DownloadState {
        let file_download_states = self.get_file_download_states().await;

        tracing::debug!(
            "[MODEL] reduce_state: model={}, file_states count={}",
            self.identifier,
            file_download_states.len()
        );
        for (i, state) in file_download_states.iter().enumerate() {
            tracing::debug!(
                "[MODEL] File {} - phase={:?}, downloaded={}/{} bytes",
                i,
                state.phase,
                state.downloaded_bytes,
                state.total_bytes
            );
        }

        let result = reduce_file_download_states(&file_download_states);

        tracing::debug!(
            "[MODEL] reduce_state result: phase={:?}, bytes={}/{}, progress={:.1}%",
            result.phase,
            result.downloaded_bytes,
            result.total_bytes,
            result.progress() * 100.0
        );

        result
    }

    pub async fn update_state_and_broadcast(
        &self,
        new_state: DownloadState,
    ) {
        tracing::debug!(
            "[MODEL] Broadcasting state: id={}, phase={:?}, progress={:.1}%",
            self.identifier,
            new_state.phase,
            new_state.progress() * 100.0
        );

        let mut state_guard = self.download_state.lock().await;
        *state_guard = new_state.clone();
        drop(state_guard);

        let own_receiver_count = self.broadcast_sender.receiver_count();
        let storage_receiver_count = self.storage_broadcast_sender.receiver_count();

        if own_receiver_count == 0 && storage_receiver_count == 0 {
            tracing::debug!("[MODEL] No receivers active, skipping broadcasts (likely shutdown)");
            return;
        }

        if own_receiver_count > 0 {
            let own_result = self.broadcast_sender.send(new_state.clone());

            match own_result {
                Ok(count) => tracing::debug!("[MODEL] ✓ Own broadcast sent to {} receiver(s)", count),
                Err(_) => {
                    tracing::trace!("[MODEL] Own broadcast skipped (no receivers)")
                },
            }
        } else {
            tracing::trace!("[MODEL] Skipping own broadcast (no receivers)");
        }

        if storage_receiver_count > 0 {
            let storage_result = self.storage_broadcast_sender.send((self.identifier.clone(), new_state));

            match storage_result {
                Ok(count) => tracing::debug!("[MODEL] ✓ Storage broadcast sent to {} receiver(s)", count),
                Err(_) => tracing::debug!("[MODEL] ⚠️ Storage broadcast failed (no receivers)"),
            }
        } else {
            tracing::trace!("[MODEL] Skipping storage broadcast (no receivers)");
        }
    }

    pub async fn file_task_by_download_id(
        &self,
        download_id: uuid::Uuid,
    ) -> Option<Arc<dyn FileDownloadTask>> {
        let file_tasks_guard = self.file_download_tasks.lock().await;
        file_tasks_guard.iter().find(|task| task.download_id() == download_id).cloned()
    }

    pub async fn reconcile(&self) -> Result<(), Error> {
        // Reconciliation is now handled by FileDownloadTask
        // Just update our broadcast with current state
        let calculated_state = self.reduce_state().await;
        self.update_state_and_broadcast(calculated_state).await;
        Ok(())
    }

    async fn ensure_file_tasks(&self) -> Result<bool, Error> {
        let mut file_tasks_guard = self.file_download_tasks.lock().await;

        // If file tasks already exist, skip creation
        if !file_tasks_guard.is_empty() {
            return Ok(false);
        }

        // Initialize file download tasks
        let mut new_file_tasks = Vec::new();
        for file_info in self.files.iter() {
            let file_path = self.cache_path.join(&file_info.name);
            let file_check = download_manager::FileCheck::CRC(file_info.crc32c().ok_or(Error::HashNotFound {
                identifier: self.identifier.clone(),
                name: file_info.name.clone(),
            })?);

            let file_task = self
                .file_download_manager
                .file_download_task(&file_info.url, &file_path, file_check, Some(file_info.size as u64))
                .await
                .map_err(|error| Error::DownloadManager {
                    message: error.to_string(),
                })?;

            // Start listening to global broadcast
            file_task.start_listening((*self.file_download_manager.global_broadcast_sender()).clone()).await;

            new_file_tasks.push(file_task);
        }

        *file_tasks_guard = new_file_tasks;
        Ok(true)
    }

    pub async fn download(&self) -> Result<(), Error> {
        tracing::debug!("[MODEL] download() called for model: {}", self.identifier);

        // Ensure file tasks are initialized
        let tasks_were_created = self.ensure_file_tasks().await?;

        // Only restart listener if file tasks were just created
        if tasks_were_created {
            tracing::debug!("[MODEL] Restarting listener for model: {} (file tasks created)", self.identifier);
            self.stop_listening().await;
            self.start_listening().await;
        }

        // Reduce state after ensuring file tasks
        let current_state = self.reduce_state().await;
        self.update_state_and_broadcast(current_state.clone()).await;

        tracing::debug!(
            "[MODEL] Current state: phase={:?}, can_transition={}",
            current_state.phase,
            Self::can_transition_to_downloading(&current_state.phase)
        );

        if !Self::can_transition_to_downloading(&current_state.phase) {
            return Err(Error::InvalidStateTransition {
                from: current_state.phase.clone(),
                to: DownloadPhase::Downloading {},
            });
        }

        tracing::debug!("[MODEL] Calling ensure_downloading");

        let result = self.ensure_downloading().await;

        tracing::debug!("[MODEL] download() completed for model: {}", self.identifier);

        result
    }

    pub async fn pause(&self) -> Result<(), Error> {
        tracing::debug!("[MODEL] pause() called for model: {}", self.identifier);

        // Ensure file tasks are initialized
        let tasks_were_created = self.ensure_file_tasks().await?;

        // Only restart listener if file tasks were just created
        if tasks_were_created {
            tracing::debug!("[MODEL] Restarting listener for model: {} (file tasks created)", self.identifier);
            self.stop_listening().await;
            self.start_listening().await;
        }

        // Reduce state after ensuring file tasks
        let current_state = self.reduce_state().await;
        self.update_state_and_broadcast(current_state.clone()).await;

        if !current_state.can_pause() {
            return Err(Error::InvalidStateTransition {
                from: current_state.phase.clone(),
                to: DownloadPhase::Paused {},
            });
        }

        tracing::debug!("[MODEL] Calling ensure_paused");

        let result = self.ensure_paused().await;

        tracing::debug!("[MODEL] pause() completed for model: {}", self.identifier);

        result
    }

    pub async fn cancel(&self) -> Result<(), Error> {
        // Ensure file tasks are initialized
        let tasks_were_created = self.ensure_file_tasks().await?;

        // Only restart listener if file tasks were just created
        if tasks_were_created {
            tracing::debug!("[MODEL] Restarting listener for model: {} (file tasks created)", self.identifier);
            self.stop_listening().await;
            self.start_listening().await;
        }

        // Reduce state after ensuring file tasks
        let current_state = self.reduce_state().await;
        self.update_state_and_broadcast(current_state.clone()).await;

        let file_tasks_guard = self.file_download_tasks.lock().await;
        for file_task in file_tasks_guard.iter() {
            let _ = file_task.cancel().await;
        }
        drop(file_tasks_guard);

        for file_info in self.files.iter() {
            let file_path = self.cache_path.join(&file_info.name);
            if file_path.exists() {
                let _ = remove_file(&file_path);
            }
            let resume_data_path = format!("{}.resume_data", file_path.display());
            if Path::new(&resume_data_path).exists() {
                let _ = remove_file(&resume_data_path);
            }
            if let Some(filename) = file_path.file_name().and_then(|n| n.to_str()) {
                let _ = CrcSnapshot::remove_crc(&self.cache_path, filename);
            }
        }

        if self.cache_path.exists() {
            let _ = remove_dir_all(&self.cache_path);
        }

        let total_bytes: u64 = self.files.iter().map(|f| f.size as u64).sum();
        let not_downloaded_state = DownloadState::not_downloaded(total_bytes as i64);
        self.update_state_and_broadcast(not_downloaded_state).await;
        Ok(())
    }

    pub async fn progress(&self) -> Result<BroadcastStream<DownloadState>, Error> {
        Ok(BroadcastStream::new(self.broadcast_sender.subscribe()))
    }

    /// Handle file task state update
    /// Called by ModelStorage listener when a file task broadcasts a state change
    pub async fn handle_file_task_update(&self) {
        tracing::debug!("[MODEL] handle_file_task_update: Computing new state for model: {}", self.identifier);

        let new_state = self.reduce_state().await;

        tracing::debug!(
            "[MODEL] handle_file_task_update: New state computed: phase={:?}, bytes={}/{}, progress={:.1}%",
            new_state.phase,
            new_state.downloaded_bytes,
            new_state.total_bytes,
            new_state.progress() * 100.0
        );

        self.update_state_and_broadcast(new_state).await;
    }

    async fn ensure_downloading(&self) -> Result<(), Error> {
        tracing::debug!("[MODEL] ensure_downloading: model={}", self.identifier);

        create_dir_all(&self.cache_path).map_err(|error| Error::IO {
            message: error.to_string(),
        })?;

        let file_tasks_guard = self.file_download_tasks.lock().await;
        let download_futures = file_tasks_guard.iter().map(|file_task| {
            let file_task = file_task.clone();
            async move {
                let _ = file_task.download().await;
            }
        });
        join_all(download_futures).await;

        tracing::debug!("[MODEL] ensure_downloading: All file downloads initiated");
        Ok(())
    }

    async fn ensure_paused(&self) -> Result<(), Error> {
        tracing::debug!("[MODEL] ensure_paused: model={}", self.identifier);
        let file_tasks_guard = self.file_download_tasks.lock().await;
        let pause_futures = file_tasks_guard.iter().map(|file_task| {
            let file_task = file_task.clone();
            async move {
                let _ = file_task.pause().await;
            }
        });
        join_all(pause_futures).await;

        tracing::debug!("[MODEL] ensure_paused: All file tasks paused");
        Ok(())
    }

    #[allow(dead_code)]
    async fn cleanup_empty_directory(&self) -> Result<(), Error> {
        if !self.cache_path.exists() {
            return Ok(());
        }
        let entries = read_dir(&self.cache_path).map_err(|error| Error::IO {
            message: error.to_string(),
        })?;
        let mut has_real_files = false;
        for entry in entries {
            if let Ok(entry) = entry {
                if let Some(name) = entry.file_name().to_str() {
                    if !name.ends_with(".resume_data") && !name.ends_with(".crc") && !name.starts_with('.') {
                        has_real_files = true;
                        break;
                    }
                }
            }
        }
        if !has_real_files {
            let _ = remove_dir_all(&self.cache_path);
        }
        Ok(())
    }

    /// Start listening to file task broadcasts
    pub async fn start_listening(&self) {
        // Check if already listening
        let mut listener_guard = self.listener_task.lock().await;
        if listener_guard.is_some() {
            tracing::debug!("[MODEL] start_listening: Already listening for model: {}", self.identifier);
            return;
        }

        // Collect all file task broadcast streams and initialize cache
        let file_tasks_guard = self.file_download_tasks.lock().await;
        let num_files = file_tasks_guard.len();
        let mut streams = Vec::new();
        let mut initial_states = Vec::new();
        for (idx, file_task) in file_tasks_guard.iter().enumerate() {
            let sender = file_task.broadcast_sender();
            let receiver_count_before = sender.receiver_count();
            let stream = BroadcastStream::new(sender.subscribe());
            tracing::info!(
                "[MODEL] Subscribed to file task {}: model={}, receivers_before={}, receivers_after={}",
                idx,
                self.identifier,
                receiver_count_before,
                sender.receiver_count()
            );
            streams.push((idx, stream));

            let mut state = file_task.state().await;
            if state.total_bytes == 0 {
                if let Some(file_info) = self.files.get(idx) {
                    if file_info.size > 0 {
                        state.total_bytes = file_info.size as u64;
                    }
                }
            }
            initial_states.push(state);
        }
        drop(file_tasks_guard);

        if streams.is_empty() {
            tracing::debug!("[MODEL] start_listening: No file tasks yet for model: {}", self.identifier);
            return;
        }

        // Pre-allocate cache with correct size BEFORE starting listener
        // This ensures all updates will have valid indices
        {
            let mut cache_guard = self.file_download_states.lock().await;
            *cache_guard = initial_states;

            debug_assert_eq!(cache_guard.len(), num_files, "Cache size must match number of file tasks");

            tracing::debug!(
                "[MODEL] Cache initialized with {} entries for model: {}",
                cache_guard.len(),
                self.identifier
            );
        }

        tracing::debug!(
            "[MODEL] start_listening: Starting listener for model: {} ({} streams)",
            self.identifier,
            streams.len()
        );

        let model = self.clone();
        tracing::debug!("[MODEL] Spawning listener task for model: {}", self.identifier);
        let handle = self.handle.spawn(async move {
            tracing::debug!("[MODEL] Listener task started for model: {}", model.identifier);
            use tokio_stream::StreamExt as TokioStreamExt;

            // Fan-in: per-stream forwarders into a single bounded channel
            let num_streams = streams.len();
            let (tx, mut rx) = tokio::sync::mpsc::channel::<(usize, FileDownloadState)>(1024);

            for (idx, mut stream) in streams {
                let tx = tx.clone();
                model.handle.spawn(async move {
                    while let Some(item) = stream.next().await {
                        match item {
                            Ok(state) => {
                                // Forward latest state; drop if aggregator has gone away
                                let _ = tx.send((idx, state)).await;
                            },
                            Err(_e) => {
                                // Lag on a single stream isn't fatal
                            },
                        }
                    }
                });
            }
            drop(tx); // close when all forwarders end

            // Aggregator: coalesce bursts, keep only latest state per task
            let mut pending: Vec<Option<FileDownloadState>> = vec![None; num_streams];

            while let Some((idx, state)) = rx.recv().await {
                pending[idx] = Some(state);

                // Drain quickly to coalesce into latest-only
                while let Ok((i, s)) = rx.try_recv() {
                    pending[i] = Some(s);
                }

                // Apply all latest states atomically, then notify once
                {
                    let mut cache_guard = model.file_download_states.lock().await;
                    for (i, slot) in pending.iter_mut().enumerate() {
                        if let Some(mut s) = slot.take() {
                            if i < cache_guard.len() {
                                // Patch total bytes if 0 in incoming update
                                if s.total_bytes == 0 {
                                    if let Some(file_info) = model.files.get(i) {
                                        if file_info.size > 0 {
                                            s.total_bytes = file_info.size as u64;
                                        }
                                    }
                                }
                                cache_guard[i] = s;
                            } else {
                                tracing::error!(
                                    "[MODEL] CRITICAL: File task index {} out of bounds (cache size: {}), model={}",
                                    i,
                                    cache_guard.len(),
                                    model.identifier
                                );
                            }
                        }
                    }
                }

                model.handle_file_task_update().await;
            }

            tracing::debug!("[MODEL] All streams ended for model: {}", model.identifier);

            tracing::debug!("[MODEL] Listener task ended for model: {}", model.identifier);
        });

        *listener_guard = Some(handle);
    }

    /// Stop listening to file task broadcasts
    pub async fn stop_listening(&self) {
        let mut listener_guard = self.listener_task.lock().await;
        if let Some(handle) = listener_guard.take() {
            handle.abort();
            let _ = handle.await;
        }
    }

    fn can_transition_to_downloading(from: &DownloadPhase) -> bool {
        matches!(
            from,
            DownloadPhase::NotDownloaded {}
                | DownloadPhase::Downloading {}
                | DownloadPhase::Paused {}
                | DownloadPhase::Error { .. }
        )
    }
}
