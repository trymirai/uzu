use std::path::{Path, PathBuf};

use async_shutdown::ShutdownManager;
use tokio::task::JoinHandle as TokioJoinHandle;

use crate::{
    backends::universal::UniversalBackend,
    traits::{ActiveTask, CancelOutcome, DownloadBackend},
};

pub struct UniversalActiveTask {
    shutdown_manager: ShutdownManager<()>,
    task_handles: Box<[TokioJoinHandle<()>]>,
    resume_artifact_path: PathBuf,
}

impl std::fmt::Debug for UniversalActiveTask {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter
            .debug_struct("UniversalActiveTask")
            .field("resume_artifact_path", &self.resume_artifact_path)
            .finish_non_exhaustive()
    }
}

impl UniversalActiveTask {
    pub fn new(
        shutdown_manager: ShutdownManager<()>,
        task_handles: Box<[TokioJoinHandle<()>]>,
        resume_artifact_path: PathBuf,
    ) -> Self {
        Self {
            shutdown_manager,
            task_handles,
            resume_artifact_path,
        }
    }
}

#[async_trait::async_trait]
impl ActiveTask for UniversalActiveTask {
    type Backend = UniversalBackend;

    async fn pause(
        self,
        _destination: &Path,
    ) -> Result<PathBuf, <Self::Backend as DownloadBackend>::Error> {
        let _ = self.shutdown_manager.trigger_shutdown(());
        for task_handle in self.task_handles {
            task_handle.abort();
            let _ = task_handle.await;
        }
        Ok(self.resume_artifact_path)
    }

    async fn cancel(
        self,
        _destination: &Path,
    ) -> CancelOutcome {
        let _ = self.shutdown_manager.trigger_shutdown(());
        for task_handle in self.task_handles {
            task_handle.abort();
            let _ = task_handle.await;
        }
        CancelOutcome::BestEffort
    }
}
