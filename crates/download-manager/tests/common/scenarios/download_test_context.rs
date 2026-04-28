use std::{path::PathBuf, sync::Arc, time::Duration};

use download_manager::{FileCheck, FileDownloadPhase, FileDownloadState, FileDownloadTask};
use tokio::{fs::read as tokio_read, time::timeout as tokio_timeout};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};
use uuid::Uuid;

use crate::common::{
    mock_download_server::{FilePayload, MockDownloadServer, RegistryFixture, RouteBehavior},
    scenarios::PhaseKind,
};

pub struct DownloadTestContext {
    _server: MockDownloadServer,
    pub payload: FilePayload,
    pub destination: PathBuf,
    _temp_dir_guard: tempfile::TempDir,
}

impl DownloadTestContext {
    pub async fn new(
        file_name: &str,
        behavior: RouteBehavior,
    ) -> Self {
        let server = MockDownloadServer::start().await;
        let path_prefix = format!("download-manager/{}", Uuid::new_v4());
        let fixture = RegistryFixture::llama_3_2_1b_instruct(&server.base_url(), &path_prefix);
        let payload = fixture.payload(file_name);
        server.serve_file(payload.clone(), behavior).await;

        let temp_dir_guard = tempfile::tempdir().unwrap();
        let destination = temp_dir_guard.path().join(&payload.file.name);

        Self {
            _server: server,
            payload,
            destination,
            _temp_dir_guard: temp_dir_guard,
        }
    }

    pub fn file_check(&self) -> FileCheck {
        FileCheck::CRC(self.payload.crc32c())
    }

    pub fn file_size(&self) -> Option<u64> {
        Some(self.payload.file.size as u64)
    }

    pub async fn assert_downloaded(
        &self,
        state: &FileDownloadState,
    ) {
        assert!(matches!(state.phase, FileDownloadPhase::Downloaded));
        assert_eq!(state.downloaded_bytes, self.payload.file.size as u64);
        assert_eq!(state.total_bytes, self.payload.file.size as u64);
        assert_eq!(tokio_read(&self.destination).await.unwrap(), self.payload.bytes.to_vec());
        assert!(
            PathBuf::from(format!("{}.crc", self.destination.display())).exists(),
            "CRC sidecar should exist after a checked download"
        );
    }

    pub fn lock_path(&self) -> PathBuf {
        PathBuf::from(format!("{}.lock", self.destination.display()))
    }
}

pub async fn wait_for_phase_kind(
    task: &Arc<dyn FileDownloadTask>,
    progress_stream: &mut BroadcastStream<FileDownloadState>,
    phase_kind: PhaseKind,
) -> FileDownloadState {
    tokio_timeout(Duration::from_secs(15), async {
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
    .unwrap()
}

pub fn error_message(state: FileDownloadState) -> String {
    let FileDownloadPhase::Error(message) = state.phase else {
        unreachable!()
    };
    message
}
