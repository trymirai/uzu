#![allow(dead_code, unused_imports)]

use std::{
    sync::{Arc, Once},
    time::Duration,
};

use download_manager::{FileDownloadPhase, FileDownloadState, FileDownloadTask};
pub use mock_registry::{Behavior, MockRegistry};
use tokio::time::timeout;
use tokio_stream::{StreamExt, wrappers::BroadcastStream};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

static INIT_TRACING: Once = Once::new();

pub fn init_test_tracing() {
    INIT_TRACING.call_once(|| {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let log_directory =
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")).join("tests").join("logs");
        std::fs::create_dir_all(&log_directory).ok();
        let file_appender =
            tracing_appender::rolling::never(&log_directory, format!("download_manager_{timestamp}.log"));
        let (non_blocking_writer, guard) = tracing_appender::non_blocking(file_appender);
        std::mem::forget(guard);

        tracing_subscriber::registry()
            .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")))
            .with(fmt::layer().with_writer(non_blocking_writer).with_ansi(false))
            .init();
        tracing::info!("download-manager test logging initialized");
    });
}

pub async fn wait_for_phase(
    task: &Arc<dyn FileDownloadTask>,
    progress_stream: &mut BroadcastStream<FileDownloadState>,
    mut is_expected_phase: impl FnMut(&FileDownloadPhase) -> bool,
) -> FileDownloadState {
    init_test_tracing();
    timeout(Duration::from_secs(15), async {
        let state = task.state().await;
        tracing::debug!(?state, "observed initial file download state");
        if is_expected_phase(&state.phase) {
            return state;
        }

        while let Some(result) = progress_stream.next().await {
            let state = result.expect("download progress stream must not lag");
            tracing::debug!(?state, "observed file download progress state");
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
