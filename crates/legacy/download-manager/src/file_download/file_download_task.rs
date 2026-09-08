use std::sync::Arc;

use kiban::stream::BoxStream;
use tokio::sync::{
    mpsc::Sender as TokioMpscSender,
    oneshot::{Sender as TokioOneshotSender, channel as tokio_oneshot_channel},
    watch::Receiver as TokioWatchReceiver,
};
use tokio_stream::wrappers::WatchStream;

use crate::{
    DownloadError, DownloadState, DownloadTaskRequest,
    file_download::{Command, DownloadConfig},
};

pub struct FileDownloadTask {
    pub request: DownloadTaskRequest,
    config: Arc<DownloadConfig>,
    commands: TokioMpscSender<(Command, TokioOneshotSender<Result<(), DownloadError>>)>,
    state: TokioWatchReceiver<DownloadState>,
}

impl FileDownloadTask {
    pub fn new(
        request: DownloadTaskRequest,
        config: Arc<DownloadConfig>,
        commands: TokioMpscSender<(Command, TokioOneshotSender<Result<(), DownloadError>>)>,
        state: TokioWatchReceiver<DownloadState>,
    ) -> Self {
        Self {
            request,
            config,
            commands,
            state,
        }
    }

    pub fn state(&self) -> DownloadState {
        self.state.borrow().clone()
    }

    pub fn live_state(&self) -> TokioWatchReceiver<DownloadState> {
        self.state.clone()
    }

    pub fn progress(&self) -> BoxStream<'static, DownloadState> {
        Box::pin(WatchStream::from_changes(self.state.clone()))
    }

    pub async fn download(&self) -> Result<(), DownloadError> {
        self.send(Command::Download).await
    }

    pub async fn pause(&self) -> Result<(), DownloadError> {
        self.send(Command::Pause).await
    }

    pub async fn delete(&self) -> Result<(), DownloadError> {
        self.send(Command::Delete).await
    }

    pub async fn foreign_owner(&self) -> Option<String> {
        self.config.foreign_owner().await
    }

    async fn send(
        &self,
        command: Command,
    ) -> Result<(), DownloadError> {
        let (reply, response) = tokio_oneshot_channel();
        self.commands.send((command, reply)).await.map_err(|_| DownloadError::ChannelClosed)?;
        response.await.unwrap_or(Err(DownloadError::ChannelClosed))
    }
}
