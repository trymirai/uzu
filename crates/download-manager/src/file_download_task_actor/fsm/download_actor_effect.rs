use std::path::PathBuf;

use crate::{
    DownloadError, FileDownloadEvent,
    file_download_task_actor::{ProgressCounters, PublicProjection, TerminalOutcome},
    traits::DownloadBackend,
};

pub enum DownloadActorEffect<B: DownloadBackend> {
    SetProjection(PublicProjection),
    SetProgress(ProgressCounters),
    EmitGlobalEvent(FileDownloadEvent),
    CompleteWaiters(TerminalOutcome),
    Reply(Result<(), DownloadError>),
    LogFsmTransition {
        from: &'static str,
        to: &'static str,
    },
    AttachActiveTask(B::ActiveTask),
    DeleteFile {
        path: PathBuf,
    },
    DeleteResumeArtifacts {
        destination: PathBuf,
    },
}
