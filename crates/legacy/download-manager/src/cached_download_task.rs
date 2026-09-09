use std::sync::Weak;

use tokio::sync::watch::Receiver as TokioWatchReceiver;

use crate::{DownloadState, DownloadTask};

pub struct CachedDownloadTask {
    pub task: Weak<DownloadTask>,
    pub live_state: Option<TokioWatchReceiver<DownloadState>>,
}
