use tokio::sync::oneshot::Sender as TokioOneshotSender;

use crate::{DownloadError, file_download_task::InactiveTaskShutdown};

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
    CancelAndDelete {
        reply_sender: TokioOneshotSender<Result<(), DownloadError>>,
    },
    Remove {
        reply_sender: TokioOneshotSender<Result<(), DownloadError>>,
    },
    RemoveIfInactive {
        reply_sender: TokioOneshotSender<Result<InactiveTaskShutdown, DownloadError>>,
    },
    StopPreservingArtifactsIfInactive {
        reply_sender: TokioOneshotSender<Result<InactiveTaskShutdown, DownloadError>>,
    },
}
