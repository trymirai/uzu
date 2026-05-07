use crate::{
    DownloadError,
    file_download_task_actor::{ProgressCounters, PublicProjection, TerminalOutcome},
};

pub enum DownloadActorEffect {
    SetProjection(PublicProjection),
    SetProgress(ProgressCounters),
    CompleteWaiters(TerminalOutcome),
    Reply(Result<(), DownloadError>),
    LogFsmTransition {
        from: &'static str,
        to: &'static str,
    },
}
