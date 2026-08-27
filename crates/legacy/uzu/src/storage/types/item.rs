use std::{
    fmt::{Debug, Formatter, Result as FmtResult},
    fs::{create_dir_all, remove_dir_all},
    path::{Path, PathBuf},
    sync::Arc,
};

use download_manager::{
    DownloadError, FileDownloadManager, FileDownloadPhase, FileDownloadState, FileDownloadTask, HttpDownloadRequest,
};
use futures_util::future::join_all;
use kiban::rt::{RuntimeHandle, TaskJoinHandle};
use tokio::sync::{
    broadcast::{Sender as TokioBroadcastSender, channel as tokio_broadcast_channel},
    mpsc::channel as tokio_mpsc_channel,
};
use tokio_stream::{StreamExt as TokioStreamExt, wrappers::BroadcastStream as TokioBroadcastStream};
use uuid::Uuid;

use crate::{
    helpers::SharedAccess,
    models::ResolvedFile,
    storage::{
        StorageError,
        types::{DownloadPhase, DownloadState, StorageDownloadEventSender, reduce_file_download_states},
    },
};

pub struct Item {
    pub identifier: String,
    files: Arc<Vec<ResolvedFile>>,
    pub cache_path: PathBuf,

    download_state: SharedAccess<DownloadState>,
    file_download_manager: Arc<dyn FileDownloadManager>,
    file_download_tasks: SharedAccess<Vec<Arc<dyn FileDownloadTask>>>,
    file_download_states: SharedAccess<Vec<FileDownloadState>>,

    runtime_handle: RuntimeHandle,
    broadcast_sender: TokioBroadcastSender<DownloadState>,
    storage_broadcast_sender: StorageDownloadEventSender,
    listener_task: SharedAccess<Option<Box<dyn TaskJoinHandle<()>>>>,
}

impl Debug for Item {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> FmtResult {
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
            runtime_handle: self.runtime_handle.clone(),
            broadcast_sender: self.broadcast_sender.clone(),
            storage_broadcast_sender: self.storage_broadcast_sender.clone(),
            listener_task: self.listener_task.clone(),
        }
    }
}

impl Item {
    pub fn matches(
        &self,
        cache_path: &Path,
        files: &[ResolvedFile],
    ) -> bool {
        self.cache_path == cache_path && self.files.as_slice() == files
    }

    pub fn new(
        identifier: String,
        files: Arc<Vec<ResolvedFile>>,
        cache_path: PathBuf,
        download_state: DownloadState,
        file_download_manager: Arc<dyn FileDownloadManager>,
        runtime_handle: RuntimeHandle,
        storage_broadcast_sender: StorageDownloadEventSender,
    ) -> Self {
        let (broadcast_sender, _) = tokio_broadcast_channel(64);
        let file_download_states = SharedAccess::new(Vec::new());
        Self {
            identifier,
            files,
            cache_path,
            download_state: SharedAccess::new(download_state),
            file_download_manager,
            file_download_tasks: SharedAccess::new(Vec::new()),
            file_download_states,
            runtime_handle,
            broadcast_sender,
            storage_broadcast_sender,
            listener_task: SharedAccess::new(None),
        }
    }

    pub async fn state(&self) -> DownloadState {
        self.download_state.lock().await.clone()
    }

    async fn get_file_download_states(&self) -> Vec<FileDownloadState> {
        let files = Arc::clone(&self.files);
        let file_tasks_guard = self.file_download_tasks.lock().await;
        let num_file_tasks = file_tasks_guard.len();
        drop(file_tasks_guard);

        let cache_guard = self.file_download_states.lock().await;

        if !cache_guard.is_empty() && cache_guard.len() == num_file_tasks {
            let mut states = cache_guard.clone();
            for (state, file) in states.iter_mut().zip(files.iter()) {
                Self::fill_total_bytes(state, file);
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

        let file_tasks = self.file_download_tasks.lock().await.clone();
        let mut states = Vec::new();
        for file_task in file_tasks {
            states.push(file_task.state().await);
        }

        for (state, file) in states.iter_mut().zip(files.iter()) {
            Self::fill_total_bytes(state, file);
        }

        states
    }

    pub async fn reduce_state(&self) -> DownloadState {
        let file_download_states = self.get_file_download_states().await;
        reduce_file_download_states(&file_download_states)
    }

    pub async fn update_state_and_broadcast(
        &self,
        new_state: DownloadState,
    ) {
        *self.download_state.lock().await = new_state.clone();

        let _ = self.broadcast_sender.send(new_state.clone());
        let _ = self.storage_broadcast_sender.send((self.identifier.clone(), new_state));
    }

    pub async fn file_task_by_download_id(
        &self,
        download_id: Uuid,
    ) -> Option<Arc<dyn FileDownloadTask>> {
        let file_tasks_guard = self.file_download_tasks.lock().await;
        file_tasks_guard.iter().find(|task| task.download_id() == download_id).cloned()
    }

    pub async fn reconcile(&self) -> Result<(), StorageError> {
        let calculated_state = self.reduce_state().await;
        self.update_state_and_broadcast(calculated_state).await;
        Ok(())
    }

    pub async fn ensure_file_tasks(
        &self,
        huggingface_api_key: Option<&Arc<str>>,
    ) -> Result<bool, StorageError> {
        if !self.file_download_tasks.lock().await.is_empty() {
            return Ok(false);
        }

        let mut new_file_tasks = Vec::with_capacity(self.files.len());
        for file_info in self.files.iter() {
            let file_path = self.cache_path.join(&file_info.file.name);
            let expected_bytes = u64::try_from(file_info.file.size).map_err(|_| StorageError::IO {
                message: format!("invalid file size for {}/{}", self.identifier, file_info.file.name),
            })?;
            let request = if file_info.requires_authentication {
                let api_key = huggingface_api_key.ok_or_else(|| StorageError::DownloadManager {
                    message: "Hugging Face authentication is required".to_string(),
                })?;
                HttpDownloadRequest::with_bearer_token(file_info.file.url.clone(), api_key)
            } else {
                HttpDownloadRequest::get(file_info.file.url.clone())
            };

            let file_task = self
                .file_download_manager
                .file_download_task(request, &file_path, file_info.check.clone(), Some(expected_bytes))
                .await
                .map_err(|error| StorageError::DownloadManager {
                    message: error.to_string(),
                })?;

            file_task.start_listening((*self.file_download_manager.global_broadcast_sender()).clone()).await;

            new_file_tasks.push(file_task);
        }

        let mut file_tasks = self.file_download_tasks.lock().await;
        if file_tasks.is_empty() {
            *file_tasks = new_file_tasks;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub async fn download(
        &self,
        huggingface_api_key: Option<Arc<str>>,
    ) -> Result<(), StorageError> {
        let tasks_were_created = self.ensure_file_tasks(huggingface_api_key.as_ref()).await?;
        if tasks_were_created {
            self.stop_listening().await;
            self.start_listening().await;
        }

        let current_state = self.reduce_state().await;
        if matches!(current_state.phase, DownloadPhase::Downloaded {}) {
            self.update_state_and_broadcast(current_state).await;
            return Ok(());
        }
        if !Self::can_transition_to_downloading(&current_state.phase) {
            return Err(StorageError::InvalidStateTransition {
                from: current_state.phase.clone(),
                to: DownloadPhase::Downloading {},
            });
        }

        self.ensure_downloading().await?;

        self.stop_listening().await;
        self.start_listening().await;

        let downloading_state = self.reduce_state().await;
        self.update_state_and_broadcast(downloading_state).await;

        Ok(())
    }

    pub async fn pause(&self) -> Result<(), StorageError> {
        let current_state = self.reduce_state().await;
        if !current_state.can_pause() {
            return Err(StorageError::InvalidStateTransition {
                from: current_state.phase.clone(),
                to: DownloadPhase::Paused {},
            });
        }

        self.ensure_paused().await?;

        let file_tasks = self.file_download_tasks.lock().await.clone();
        let mut file_download_states = Vec::with_capacity(file_tasks.len());
        for file_task in file_tasks {
            file_download_states.push(file_task.state().await);
        }

        *self.file_download_states.lock().await = file_download_states;
        let paused_state = self.reduce_state().await;
        self.update_state_and_broadcast(paused_state).await;

        Ok(())
    }

    pub async fn cancel(&self) -> Result<(), StorageError> {
        let files = Arc::clone(&self.files);
        for file_info in files.iter() {
            let destination = self.cache_path.join(&file_info.file.name);
            if let Some(owner) = self.file_download_manager.destination_foreign_lock(&destination).await {
                tracing::info!(?destination, %owner, "refusing to cancel: destination is locked by another manager");
                return Err(StorageError::InvalidStateTransition {
                    from: DownloadPhase::Locked {},
                    to: DownloadPhase::NotDownloaded {},
                });
            }
        }

        let current_state = self.reduce_state().await;
        if matches!(current_state.phase, DownloadPhase::Locked {}) {
            return Err(StorageError::InvalidStateTransition {
                from: current_state.phase,
                to: DownloadPhase::NotDownloaded {},
            });
        }

        self.cancel_and_remove_active_file_tasks().await?;

        self.stop_listening().await;
        *self.file_download_tasks.lock().await = Vec::new();
        *self.file_download_states.lock().await = Vec::new();

        if self.cache_path.exists() {
            let _ = remove_dir_all(&self.cache_path);
        }

        let not_downloaded_state = DownloadState::not_downloaded(Self::total_bytes(&files)?);
        self.update_state_and_broadcast(not_downloaded_state).await;
        Ok(())
    }

    pub async fn progress(&self) -> Result<TokioBroadcastStream<DownloadState>, StorageError> {
        Ok(TokioBroadcastStream::new(self.broadcast_sender.subscribe()))
    }

    pub async fn detach_active_downloads(&self) -> Result<(), StorageError> {
        self.cancel_and_remove_active_file_tasks().await?;
        *self.file_download_tasks.lock().await = Vec::new();
        *self.file_download_states.lock().await = Vec::new();
        Ok(())
    }

    async fn cancel_and_remove_active_file_tasks(&self) -> Result<(), StorageError> {
        let file_tasks = self.file_download_tasks.lock().await.clone();
        let cancel_futures = file_tasks.iter().map(|file_task| {
            let file_task = file_task.clone();
            let manager = self.file_download_manager.clone();
            async move {
                let download_id = file_task.download_id();
                file_task.cancel().await?;
                manager.remove_file_task(download_id).await?;
                Ok::<(), DownloadError>(())
            }
        });
        let first_error = join_all(cancel_futures).await.into_iter().find_map(Result::err);

        match first_error {
            Some(error) => Err(StorageError::DownloadManager {
                message: error.to_string(),
            }),
            None => Ok(()),
        }
    }

    /// Handle file task state update
    /// Called by ModelStorage listener when a file task broadcasts a state change
    pub async fn handle_file_task_update(&self) {
        let new_state = self.reduce_state().await;
        self.update_state_and_broadcast(new_state).await;
    }

    async fn ensure_downloading(&self) -> Result<(), StorageError> {
        create_dir_all(&self.cache_path).map_err(|error| StorageError::IO {
            message: error.to_string(),
        })?;

        let file_tasks = self.file_download_tasks.lock().await.clone();
        let download_futures = file_tasks.iter().map(|file_task| {
            let file_task = file_task.clone();
            async move { file_task.download().await }
        });
        let first_error = join_all(download_futures).await.into_iter().find_map(Result::err);
        if let Some(error) = first_error {
            return Err(StorageError::DownloadManager {
                message: error.to_string(),
            });
        }

        Ok(())
    }

    async fn ensure_paused(&self) -> Result<(), StorageError> {
        let file_tasks = self.file_download_tasks.lock().await.clone();
        let pause_futures = file_tasks.iter().map(|file_task| {
            let file_task = file_task.clone();
            async move {
                if matches!(file_task.state().await.phase, FileDownloadPhase::Downloading) {
                    match file_task.pause().await {
                        Ok(()) => Ok(()),
                        Err(DownloadError::InvalidStateTransition)
                            if !matches!(file_task.state().await.phase, FileDownloadPhase::Downloading) =>
                        {
                            Ok(())
                        },
                        Err(error) => Err(error),
                    }
                } else {
                    Ok(())
                }
            }
        });
        let first_error = join_all(pause_futures).await.into_iter().find_map(Result::err);
        if let Some(error) = first_error {
            return Err(StorageError::DownloadManager {
                message: error.to_string(),
            });
        }

        Ok(())
    }

    /// Start listening to file task broadcasts
    pub async fn start_listening(&self) {
        if self.listener_task.lock().await.is_some() {
            return;
        }

        let file_tasks = self.file_download_tasks.lock().await.clone();
        let files = Arc::clone(&self.files);
        let num_files = file_tasks.len();
        let mut streams = Vec::new();
        let mut initial_states = Vec::new();
        for (idx, file_task) in file_tasks.iter().enumerate() {
            let sender = file_task.broadcast_sender();
            let stream = TokioBroadcastStream::new(sender.subscribe());
            streams.push((idx, stream));

            let mut state = file_task.state().await;
            if let Some(file) = files.get(idx) {
                Self::fill_total_bytes(&mut state, file);
            }
            initial_states.push(state);
        }
        if streams.is_empty() {
            return;
        }

        {
            let mut cache_guard = self.file_download_states.lock().await;
            *cache_guard = initial_states;

            debug_assert_eq!(cache_guard.len(), num_files, "Cache size must match number of file tasks");
        }

        let model = self.clone();
        let handle = self.runtime_handle.spawn(async move {
            let num_streams = streams.len();
            let (tx, mut rx) = tokio_mpsc_channel::<(usize, FileDownloadState)>(1024);
            let mut forwarder_handles = Vec::with_capacity(num_streams);

            for (idx, mut stream) in streams {
                let tx = tx.clone();
                let forwarder_handle = model.runtime_handle.spawn(async move {
                    while let Some(item) = stream.next().await {
                        if let Ok(state) = item {
                            let _ = tx.send((idx, state)).await;
                        }
                    }
                });
                forwarder_handles.push(forwarder_handle);
            }
            let _forwarder_handles = ListenerForwarderHandles {
                handles: forwarder_handles,
            };
            drop(tx);

            let mut pending: Vec<Option<FileDownloadState>> = vec![None; num_streams];

            while let Some((idx, state)) = rx.recv().await {
                pending[idx] = Some(state);

                while let Ok((i, s)) = rx.try_recv() {
                    pending[i] = Some(s);
                }

                {
                    let mut cache_guard = model.file_download_states.lock().await;
                    for (i, slot) in pending.iter_mut().enumerate() {
                        if let Some(mut s) = slot.take() {
                            if i < cache_guard.len() {
                                if let Some(file) = files.get(i) {
                                    Self::fill_total_bytes(&mut s, file);
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
        });

        let mut listener = self.listener_task.lock().await;
        if listener.is_none() {
            *listener = Some(handle);
        } else {
            handle.abort();
        }
    }

    /// Stop listening to file task broadcasts
    pub async fn stop_listening(&self) {
        let handle = self.listener_task.lock().await.take();
        if let Some(handle) = handle {
            handle.abort_and_join().await;
        }
    }

    pub fn total_bytes(files: &[ResolvedFile]) -> Result<i64, StorageError> {
        files.iter().try_fold(0_i64, |total, file| {
            total.checked_add(file.file.size).ok_or_else(|| StorageError::IO {
                message: "model download size overflow".to_string(),
            })
        })
    }

    fn fill_total_bytes(
        state: &mut FileDownloadState,
        file: &ResolvedFile,
    ) {
        if state.total_bytes == 0
            && let Ok(size) = u64::try_from(file.file.size)
            && size > 0
        {
            state.total_bytes = size;
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

struct ListenerForwarderHandles {
    handles: Vec<Box<dyn TaskJoinHandle<()>>>,
}

impl Drop for ListenerForwarderHandles {
    fn drop(&mut self) {
        for handle in &self.handles {
            handle.abort();
        }
    }
}
