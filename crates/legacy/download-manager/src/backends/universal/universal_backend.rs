use std::{path::Path, sync::Arc};

use kiban::{fs, rt::RuntimeHandle};
use tokio::sync::{oneshot::channel as tokio_oneshot_channel, watch::channel as tokio_watch_channel};

use crate::{
    backends::{
        ActiveTask, Backend, BackendError, BackendEventSender, DownloadGeneration,
        universal::{UniversalActiveTask, UniversalStream},
    },
    file_download::DownloadConfig,
};

pub struct UniversalBackend {
    runtime_handle: RuntimeHandle,
}

impl UniversalBackend {
    pub fn new(runtime_handle: RuntimeHandle) -> Self {
        Self {
            runtime_handle,
        }
    }
}

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
impl Backend for UniversalBackend {
    fn name(&self) -> &'static str {
        "universal"
    }

    fn resume_artifact_extension(&self) -> &'static str {
        "part"
    }

    async fn start(
        &self,
        config: Arc<DownloadConfig>,
        generation: DownloadGeneration,
        events: BackendEventSender,
    ) -> Result<Box<dyn ActiveTask>, BackendError> {
        if let Some(parent) = config.destination.parent() {
            fs::asyn::create_dir_all(parent).await?;
        }
        let (pause_sender, pause_receiver) = tokio_watch_channel(false);
        let (completion_sender, completion_receiver) = tokio_oneshot_channel();
        let stream = UniversalStream::new(config, generation, events, pause_receiver, completion_sender);
        let task_handle = self.runtime_handle.spawn(stream.run());
        Ok(Box::new(UniversalActiveTask::new(task_handle, pause_sender, completion_receiver)))
    }

    async fn read_resume_progress(
        &self,
        resume_artifact_path: &Path,
    ) -> u64 {
        fs::asyn::file_length(resume_artifact_path).await.unwrap_or(0)
    }

    async fn has_pending_task(
        &self,
        _config: &DownloadConfig,
    ) -> Result<bool, BackendError> {
        Ok(false)
    }

    async fn attach_pending_task(
        &self,
        _config: Arc<DownloadConfig>,
        _generation: DownloadGeneration,
        _events: BackendEventSender,
    ) -> Result<Option<Box<dyn ActiveTask>>, BackendError> {
        Ok(None)
    }
}
