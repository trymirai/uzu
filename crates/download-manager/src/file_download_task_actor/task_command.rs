use tokio::sync::oneshot::Sender as TokioOneshotSender;

use crate::DownloadError;

#[derive(Debug)]
pub enum TaskCommand {
    Download {
        reply_sender: TokioOneshotSender<Result<(), DownloadError>>,
    },
    Pause {
        reply_sender: TokioOneshotSender<Result<(), DownloadError>>,
    },
    Cancel {
        reply_sender: TokioOneshotSender<Result<(), DownloadError>>,
    },
}
