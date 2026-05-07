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

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc, time::Duration};

    use crate::backends::universal::UniversalActiveTask;

    #[tokio::test(flavor = "multi_thread")]
    async fn drop_aborts_running_task_handles() {
        let sentinel = Arc::new(());
        let sentinel_for_task = Arc::clone(&sentinel);
        let handle = tokio::spawn(async move {
            let _hold = sentinel_for_task;
            std::future::pending::<()>().await;
        });

        let task_handles: Box<[_]> = vec![handle].into_boxed_slice();
        let active_task = UniversalActiveTask::new(task_handles, PathBuf::from("/tmp/uzu-active-task-drop-test"));

        tokio::task::yield_now().await;
        assert_eq!(
            Arc::strong_count(&sentinel),
            2,
            "spawned task should be holding the sentinel before active_task is dropped",
        );

        drop(active_task);

        for _attempt in 0..100 {
            if Arc::strong_count(&sentinel) == 1 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("dropping UniversalActiveTask did not abort the spawned task within 1s");
    }
}
