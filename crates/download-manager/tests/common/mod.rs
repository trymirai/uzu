#![allow(dead_code, unused_imports)]

use std::{sync::Arc, time::Duration};

use download_manager::{FileDownloadPhase, FileDownloadState, FileDownloadTask};
pub use mock_registry::{Behavior, MockRegistry};
use tokio::time::timeout;
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

pub async fn wait_for_phase(
    task: &Arc<dyn FileDownloadTask>,
    progress_stream: &mut BroadcastStream<FileDownloadState>,
    mut is_expected_phase: impl FnMut(&FileDownloadPhase) -> bool,
) -> FileDownloadState {
    timeout(Duration::from_secs(15), async {
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
    })
    .await
    .expect("timed out waiting for file download phase")
}

pub fn error_message(state: FileDownloadState) -> String {
    let FileDownloadPhase::Error(message) = state.phase else {
        unreachable!("state must be an error")
    };
    message
}
