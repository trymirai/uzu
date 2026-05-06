use crate::traits::ActiveDownloadGeneration;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FsmEvent {
    Download,
    Pause,
    Cancel,
    BackendCompleted {
        generation: ActiveDownloadGeneration,
    },
    BackendError {
        generation: ActiveDownloadGeneration,
        message: String,
    },
    ProgressUpdate {
        generation: ActiveDownloadGeneration,
        downloaded_bytes: u64,
        total_bytes: Option<u64>,
    },
}
