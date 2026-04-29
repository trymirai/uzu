use tokio::sync::oneshot::Sender as TokioOneshotSender;

use crate::{DownloadError, file_download_task_actor::fsm::DownloadActorEffect, traits::DownloadBackend};

pub struct DispatchContext<B: DownloadBackend> {
    pub effects: Vec<DownloadActorEffect<B>>,
    pub pending_reply: Option<TokioOneshotSender<Result<(), DownloadError>>>,
}

impl<B: DownloadBackend> DispatchContext<B> {
    pub fn new(pending_reply: Option<TokioOneshotSender<Result<(), DownloadError>>>) -> Self {
        Self {
            effects: Vec::new(),
            pending_reply,
        }
    }

    pub fn push(
        &mut self,
        effect: DownloadActorEffect<B>,
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
