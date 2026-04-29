use std::{marker::PhantomData, sync::Arc};

use tokio::runtime::Handle as TokioHandle;
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{
    DownloadError, DownloadEvent, DownloadLogEvent, FileCheck, FileDownloadManager, FileDownloadTask,
    backends::common::{Backend, DownloadManagerState, Startup},
    compute_download_id,
    file_download_task_actor::GenericFileDownloadTask,
    record_download_log_event,
};

#[derive(Clone, Debug)]
pub struct DownloadManager<B: Backend> {
    state: DownloadManagerState,
    context: Arc<B::Context>,
    backend: PhantomData<B>,
}

impl<B: Backend> DownloadManager<B> {
    pub fn new(manager_id: String) -> Self
    where
        B::Context: Default,
    {
        record_download_log_event(DownloadLogEvent::ManagerCreated {
            manager_id: manager_id.clone(),
        });
        Self {
            state: DownloadManagerState::with_manager_id(manager_id, TokioHandle::current()),
            context: Arc::new(B::Context::default()),
            backend: PhantomData,
        }
    }

    pub fn from_tokio_handle(tokio_handle: TokioHandle) -> Result<Self, DownloadError> {
        let context = B::create_context(tokio_handle.clone())?;
        let state = DownloadManagerState::new(B::manager_suffix(), tokio_handle);
        record_download_log_event(DownloadLogEvent::ManagerCreated {
            manager_id: state.manager_id.clone(),
        });
        Ok(Self {
            state,
            context: Arc::new(context),
            backend: PhantomData,
        })
    }
}

#[async_trait::async_trait]
impl<B: Backend> FileDownloadManager for DownloadManager<B> {
    fn manager_id(&self) -> &str {
        &self.state.manager_id
    }

    fn subscribe_to_all_downloads(&self) -> TokioBroadcastStream<DownloadEvent> {
        self.state.subscribe_to_all_downloads()
    }

    fn global_broadcast_sender(&self) -> crate::SharedDownloadEventSender {
        self.state.global_broadcast_sender()
    }

    async fn get_all_file_tasks(&self) -> Result<Vec<Arc<dyn FileDownloadTask>>, DownloadError> {
        self.state.get_all_file_tasks().await
    }

    #[allow(clippy::ptr_arg)]
    async fn file_download_task(
        &self,
        source_url: &String,
        destination_path: &std::path::Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError> {
        let download_id = compute_download_id(source_url, destination_path);
        if let Some(cached_task) = self.state.get_task(download_id).await {
            return Ok(cached_task);
        }

        let startup = Startup::observe::<B>(
            download_id,
            source_url,
            destination_path,
            file_check,
            expected_bytes,
            &self.state.manager_id,
        )?;
        startup.apply_actions().await?;
        record_download_log_event(DownloadLogEvent::StartupReconciled {
            download_id,
        });

        let task = GenericFileDownloadTask::<B>::spawn_with_initial_attachment(
            startup.config,
            Arc::clone(&self.context),
            startup.initial_lifecycle_state,
            startup.initial_projection,
            startup.initial_progress,
        )
        .await?;
        let task: Arc<dyn FileDownloadTask> = Arc::new(task);
        self.state.insert_task(download_id, Arc::clone(&task)).await;
        record_download_log_event(DownloadLogEvent::TaskSpawned {
            download_id,
        });
        Ok(task)
    }
}
