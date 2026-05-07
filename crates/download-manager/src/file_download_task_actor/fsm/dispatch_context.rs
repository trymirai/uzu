use tokio::sync::oneshot::Sender as TokioOneshotSender;

use crate::{DownloadError, file_download_task_actor::fsm::DownloadActorEffect};

pub struct DispatchContext {
    pub effects: Vec<DownloadActorEffect>,
    pub pending_reply: Option<TokioOneshotSender<Result<(), DownloadError>>>,
}

impl DispatchContext {
    pub fn new(pending_reply: Option<TokioOneshotSender<Result<(), DownloadError>>>) -> Self {
        Self {
            effects: Vec::new(),
            pending_reply,
        }
    }

    pub fn push(
        &mut self,
        effect: DownloadActorEffect,
    ) {
        self.effects.push(effect);
    }

    pub fn reply(
        &mut self,
        result: Result<(), DownloadError>,
    ) {
        self.effects.push(DownloadActorEffect::Reply(result));
    }
}
