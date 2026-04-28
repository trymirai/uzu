use std::sync::Arc;

use download_manager::{FileDownloadPhase, FileDownloadState, FileDownloadTask};
pub use mock_registry::{Behavior, MockRegistry};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

pub async fn wait_for_phase(
    task: &Arc<dyn FileDownloadTask>,
    progress_stream: &mut BroadcastStream<FileDownloadState>,
    mut is_expected_phase: impl FnMut(&FileDownloadPhase) -> bool,
) -> FileDownloadState {
    let state = task.state().await;
    if is_expected_phase(&state.phase) {
        return state;
    }

    while let Some(result) = progress_stream.next().await {
        let state = result.expect("download progress stream must not lag");
        if is_expected_phase(&state.phase) {
            return state;
        }
    }

    panic!("download progress stream ended before expected phase");
}

pub fn error_message(state: FileDownloadState) -> String {
    let FileDownloadPhase::Error(message) = state.phase else {
        unreachable!("state must be an error")
    };
    message
}
