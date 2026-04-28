use std::{collections::HashMap, sync::Arc};

use tokio::sync::broadcast::Sender as TokioBroadcastSender;
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{DownloadError, DownloadId, FileDownloadEvent, FileDownloadTask, TokioHandle, TokioMutex};

pub(crate) type TaskCache = Arc<TokioMutex<HashMap<DownloadId, Arc<dyn FileDownloadTask>>>>;

#[derive(Clone)]
pub(crate) struct DownloadManagerState {
    pub(crate) manager_id: String,
    pub(crate) global_broadcast_sender: Arc<TokioBroadcastSender<(DownloadId, FileDownloadEvent)>>,
    pub(crate) tokio_handle: TokioHandle,
    pub(crate) task_cache: TaskCache,
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
    pub(crate) fn new(
        suffix: &str,
        tokio_handle: TokioHandle,
    ) -> Self {
        let manager_id = generate_manager_id(suffix);
        let (global_broadcast_sender, _global_broadcast_receiver) =
            tokio::sync::broadcast::channel::<(DownloadId, FileDownloadEvent)>(256);
        Self {
            manager_id,
            global_broadcast_sender: Arc::new(global_broadcast_sender),
            tokio_handle,
            task_cache: Arc::new(TokioMutex::new(HashMap::new())),
        }
    }

    pub(crate) fn subscribe_to_all_downloads(&self) -> TokioBroadcastStream<(DownloadId, FileDownloadEvent)> {
        TokioBroadcastStream::new(self.global_broadcast_sender.subscribe())
    }

    pub(crate) async fn get_all_file_tasks(&self) -> Result<Vec<Arc<dyn FileDownloadTask>>, DownloadError> {
        let task_cache_guard = self.task_cache.lock().await;
        Ok(task_cache_guard.values().cloned().collect())
    }
}

fn generate_manager_id(suffix: &str) -> String {
    #[cfg(target_vendor = "apple")]
    {
        use crate::NSBundle;
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
