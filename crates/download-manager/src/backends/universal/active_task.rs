use std::path::{Path, PathBuf};

use tokio::task::JoinHandle as TokioJoinHandle;

use crate::{
    backends::universal::UniversalBackend,
    traits::{ActiveTask, CancelOutcome, DownloadBackend},
};

pub struct UniversalActiveTask {
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
        task_handles: Box<[TokioJoinHandle<()>]>,
        resume_artifact_path: PathBuf,
    ) -> Self {
        Self {
            task_handles,
            resume_artifact_path,
        }
    }
}

impl Drop for UniversalActiveTask {
    fn drop(&mut self) {
        for handle in self.task_handles.iter() {
            handle.abort();
        }
    }
}

#[async_trait::async_trait]
impl ActiveTask for UniversalActiveTask {
    type Backend = UniversalBackend;

    async fn pause(
        mut self,
        _destination: &Path,
    ) -> Result<PathBuf, <Self::Backend as DownloadBackend>::Error> {
        for task_handle in std::mem::take(&mut self.task_handles) {
            task_handle.abort();
            let _ = task_handle.await;
        }
        Ok(std::mem::take(&mut self.resume_artifact_path))
    }

    async fn cancel(
        mut self,
        _destination: &Path,
    ) -> CancelOutcome {
        for task_handle in std::mem::take(&mut self.task_handles) {
            task_handle.abort();
            let _ = task_handle.await;
        }
        CancelOutcome::BestEffort
    }
}
