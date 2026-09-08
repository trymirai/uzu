use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex, PoisonError},
};

use kiban::rt::RuntimeHandle;
use tokio::sync::Mutex as TokioMutex;
use uuid::Uuid;

use crate::{
    DownloadError, DownloadId, DownloadManagerType, DownloadTask, DownloadTaskKind, DownloadTaskRequest,
    GroupDownloadTask,
    backends::{Backend, UniversalBackend},
    cached_download_task::CachedDownloadTask,
    file_download::{DownloadConfig, FileDownloadActor},
};

pub struct DownloadManager {
    manager_id: String,
    instance_id: Uuid,
    backend: Arc<dyn Backend>,
    tasks: Mutex<HashMap<DownloadId, CachedDownloadTask>>,
    construction_locks: Mutex<HashMap<DownloadId, Arc<TokioMutex<()>>>>,
}

impl DownloadManager {
    pub fn new(
        kind: DownloadManagerType,
        runtime_handle: RuntimeHandle,
    ) -> Self {
        let backend: Arc<dyn Backend> = match kind {
            #[cfg(target_vendor = "apple")]
            DownloadManagerType::Native => Arc::new(crate::backends::AppleBackend::new(runtime_handle)),
            DownloadManagerType::Universal => Arc::new(UniversalBackend::new(runtime_handle)),
        };
        #[cfg(target_vendor = "apple")]
        let bundle_id = crate::backends::AppleBackend::bundle_identifier();
        #[cfg(not(target_vendor = "apple"))]
        let bundle_id = String::new();
        let manager_id = if bundle_id.is_empty() {
            format!("mirai.{}", backend.name())
        } else {
            format!("{bundle_id}.mirai.{}", backend.name())
        };
        tracing::debug!(%manager_id, "download manager created");
        Self {
            manager_id,
            instance_id: Uuid::new_v4(),
            backend,
            tasks: Mutex::default(),
            construction_locks: Mutex::default(),
        }
    }

    pub async fn download_task(
        &self,
        request: DownloadTaskRequest,
    ) -> Result<Arc<DownloadTask>, DownloadError> {
        self.task(request.resolved(Path::new(""))).await
    }

    async fn task(
        &self,
        request: DownloadTaskRequest,
    ) -> Result<Arc<DownloadTask>, DownloadError> {
        let download_id = request.download_id();
        if let Some(cached) = self.cached(download_id, &request) {
            return cached;
        }
        let construction_lock = self
            .construction_locks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(download_id)
            .or_default()
            .clone();
        let _construction_guard = construction_lock.lock().await;
        if let Some(cached) = self.cached(download_id, &request) {
            return cached;
        }
        let stopping = self
            .tasks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&download_id)
            .and_then(|cached| cached.live_state.clone());
        if let Some(mut live_state) = stopping {
            while live_state.changed().await.is_ok() {}
        }
        let task = Arc::new(self.build(&request).await?);
        let live_state = match &*task {
            DownloadTask::File(file) => Some(file.live_state()),
            DownloadTask::Group(_) => None,
        };
        {
            let mut tasks = self.tasks.lock().unwrap_or_else(PoisonError::into_inner);
            tasks.retain(|_, cached| !cached.is_stopped());
            tasks.insert(
                download_id,
                CachedDownloadTask {
                    task: Arc::downgrade(&task),
                    live_state,
                },
            );
        }
        self.construction_locks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|_, lock| Arc::strong_count(lock) > 1);
        Ok(task)
    }

    fn cached(
        &self,
        download_id: DownloadId,
        request: &DownloadTaskRequest,
    ) -> Option<Result<Arc<DownloadTask>, DownloadError>> {
        let task = self.tasks.lock().unwrap_or_else(PoisonError::into_inner).get(&download_id)?.task.upgrade()?;
        Some(if task.request() == request {
            Ok(task)
        } else {
            Err(DownloadError::ConflictingConfig(request.destination.display().to_string()))
        })
    }

    async fn build(
        &self,
        request: &DownloadTaskRequest,
    ) -> Result<DownloadTask, DownloadError> {
        match &request.kind {
            DownloadTaskKind::File {
                source_url,
                expected_crc32c,
                expected_bytes,
            } => {
                let config = Arc::new(DownloadConfig {
                    download_id: request.download_id(),
                    source_url: source_url.clone(),
                    destination: request.destination.clone(),
                    resume_artifact_path: self.backend.resume_artifact_path(&request.destination),
                    expected_crc32c: expected_crc32c.clone(),
                    expected_bytes: *expected_bytes,
                    manager_id: self.manager_id.clone(),
                    manager_instance_id: self.instance_id,
                });
                let (state, attach_lock) = self.backend.reconcile(&config).await?;
                tracing::debug!(
                    download_id = %config.download_id,
                    phase = ?state.download_state(&config).phase,
                    "startup reconciled"
                );
                let task =
                    FileDownloadActor::spawn(Arc::clone(&self.backend), request.clone(), config, state, attach_lock)
                        .await?;
                Ok(DownloadTask::File(task))
            },
            DownloadTaskKind::Group(subrequests) => {
                let mut children = Vec::with_capacity(subrequests.len());
                for subrequest in subrequests {
                    children.push(Box::pin(self.task(subrequest.clone())).await?);
                }
                Ok(DownloadTask::Group(GroupDownloadTask::new(request.clone(), children)))
            },
        }
    }
}
