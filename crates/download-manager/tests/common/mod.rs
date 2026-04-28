use std::{sync::Arc, time::Duration};

use download_manager::{FileDownloadPhase, FileDownloadState, FileDownloadTask};
pub use mock_registry::{Behavior, MockRegistry};
use tokio::time::timeout;
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhaseKind {
    Downloaded,
    Error,
}

impl PhaseKind {
    fn matches(
        self,
        phase: &FileDownloadPhase,
    ) -> bool {
        match self {
            Self::Downloaded => matches!(phase, FileDownloadPhase::Downloaded),
            Self::Error => matches!(phase, FileDownloadPhase::Error(_)),
        }
    }
}

pub async fn wait_for_phase_kind(
    task: &Arc<dyn FileDownloadTask>,
    progress_stream: &mut BroadcastStream<FileDownloadState>,
    phase_kind: PhaseKind,
) -> FileDownloadState {
    timeout(Duration::from_secs(30), async {
        loop {
            let state = task.state().await;
            if phase_kind.matches(&state.phase) {
                return state;
            }

            if let Some(Ok(state)) = progress_stream.next().await {
                if phase_kind.matches(&state.phase) {
                    return state;
                }
            }
        }
    })
    .await
    .expect("download did not reach expected phase")
}

pub fn error_message(state: FileDownloadState) -> String {
    let FileDownloadPhase::Error(message) = state.phase else {
        unreachable!("state must be an error")
    };
    message
}
