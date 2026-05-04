use std::{collections::HashMap, sync::Arc};

use tokio::sync::{Mutex as TokioMutex, broadcast::channel as tokio_broadcast_channel};
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{DownloadError, DownloadEvent, DownloadId, FileDownloadTask, SharedDownloadEventSender};

type TaskCache = Arc<TokioMutex<HashMap<DownloadId, Arc<dyn FileDownloadTask>>>>;
type ConstructionLocks = Arc<TokioMutex<HashMap<DownloadId, Arc<TokioMutex<()>>>>>;

#[derive(Clone)]
pub struct DownloadManagerState {
    pub manager_id: String,
    pub global_broadcast_sender: SharedDownloadEventSender,
    task_cache: TaskCache,
    construction_locks: ConstructionLocks,
}

impl std::fmt::Debug for DownloadManagerState {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter.debug_struct("DownloadManagerState").field("manager_id", &self.manager_id).finish()
    }
}

impl DownloadManagerState {
    pub fn new(suffix: &str) -> Self {
        let (global_broadcast_sender, _) = tokio_broadcast_channel::<DownloadEvent>(256);
        Self {
            manager_id: generate_manager_id(suffix),
            global_broadcast_sender: Arc::new(global_broadcast_sender),
            task_cache: Arc::new(TokioMutex::new(HashMap::new())),
            construction_locks: Arc::new(TokioMutex::new(HashMap::new())),
        }
    }

    pub fn subscribe_to_all_downloads(&self) -> TokioBroadcastStream<DownloadEvent> {
        TokioBroadcastStream::new(self.global_broadcast_sender.subscribe())
    }

    pub fn global_broadcast_sender(&self) -> SharedDownloadEventSender {
        Arc::clone(&self.global_broadcast_sender)
    }

    pub async fn get_all_file_tasks(&self) -> Result<Vec<Arc<dyn FileDownloadTask>>, DownloadError> {
        Ok(self.task_cache.lock().await.values().cloned().collect())
    }

    pub async fn get_task(
        &self,
        download_id: DownloadId,
    ) -> Option<Arc<dyn FileDownloadTask>> {
        self.task_cache.lock().await.get(&download_id).cloned()
    }

    pub async fn insert_task(
        &self,
        download_id: DownloadId,
        task: Arc<dyn FileDownloadTask>,
    ) {
        self.task_cache.lock().await.insert(download_id, task);
    }

    pub async fn remove_task(
        &self,
        download_id: DownloadId,
    ) {
        self.task_cache.lock().await.remove(&download_id);
        self.construction_locks.lock().await.remove(&download_id);
    }

    pub async fn construction_lock(
        &self,
        download_id: DownloadId,
    ) -> Arc<TokioMutex<()>> {
        let mut construction_locks = self.construction_locks.lock().await;
        construction_locks.entry(download_id).or_insert_with(|| Arc::new(TokioMutex::new(()))).clone()
    }
}

fn generate_manager_id(suffix: &str) -> String {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_foundation::NSBundle;

        let bundle_id = NSBundle::mainBundle().bundleIdentifier().unwrap_or_default().to_string();
        if bundle_id.is_empty() {
            format!("mirai.{suffix}")
        } else {
            format!("{bundle_id}.mirai.{suffix}")
        }
    }

    #[cfg(not(target_vendor = "apple"))]
    {
        format!("mirai.{suffix}")
    }
}
