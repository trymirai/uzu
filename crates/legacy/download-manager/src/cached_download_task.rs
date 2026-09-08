use std::sync::Weak;

use tokio::sync::watch::Receiver as TokioWatchReceiver;

use crate::{DownloadState, DownloadTask};

pub struct CachedDownloadTask {
    pub task: Weak<DownloadTask>,
    pub live_state: Option<TokioWatchReceiver<DownloadState>>,
}

impl CachedDownloadTask {
    pub fn is_stopped(&self) -> bool {
        self.task.strong_count() == 0
            && self.live_state.as_ref().is_none_or(|live_state| live_state.has_changed().is_err())
    }
}
