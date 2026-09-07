use std::sync::Arc;

use tokio::sync::{Mutex as TokioMutex, watch::Sender as TokioWatchSender};

use crate::file_download_task_actor::PendingProgressSlot;

#[derive(Clone, Debug)]
pub struct SharedProgressCoalescer {
    pub pending_progress: Arc<TokioMutex<PendingProgressSlot>>,
    pub actor_waker: TokioWatchSender<()>,
}
