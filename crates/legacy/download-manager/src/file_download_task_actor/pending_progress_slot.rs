use crate::file_download_task_actor::BackendProgress;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PendingProgressSlot {
    pub progress: Option<BackendProgress>,
}

impl PendingProgressSlot {
    pub fn take(&mut self) -> Option<BackendProgress> {
        self.progress.take()
    }
}
