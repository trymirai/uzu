use std::{path::PathBuf, sync::Arc, time::Duration};

use download_manager::{FileCheck, FileDownloadManagerType, FileDownloadPhase, FileDownloadState, FileDownloadTask};
use tokio::{
    fs::{read as tokio_read, write as tokio_write},
    time::{sleep as tokio_sleep, timeout as tokio_timeout},
};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};
use uuid::Uuid;

use crate::common::{
    mock_download_server::{FilePayload, MockDownloadServer, RegistryFixture, RouteBehavior},
    scenarios::PhaseKind,
};

pub struct DownloadTestContext {
    pub server: MockDownloadServer,
    pub payload: FilePayload,
    pub destination: PathBuf,
    temp_dir_guard: tempfile::TempDir,
}

impl DownloadTestContext {
    pub async fn new(
        file_name: &str,
        behavior: RouteBehavior,
    ) -> Self {
        Self::new_with_existing_part_size(file_name, behavior, None).await
    }

    pub async fn new_with_existing_part_size(
        file_name: &str,
        behavior: RouteBehavior,
        existing_part_size: Option<usize>,
    ) -> Self {
        let server = MockDownloadServer::start().await;
        let path_prefix = format!("download-manager/{}", Uuid::new_v4());
        let fixture = RegistryFixture::llama_3_2_1b_instruct(&server.base_url(), &path_prefix);
        let payload = fixture.payload(file_name);
        server.serve_file(payload.clone(), behavior).await;

        let temp_dir_guard = tempfile::tempdir().expect("failed to create test temp dir");
        let destination = temp_dir_guard.path().join(&payload.file.name);
        if let Some(existing_part_size) = existing_part_size {
            tokio_write(destination.with_extension("part"), payload.bytes[..existing_part_size].to_vec())
                .await
                .expect("failed to seed partial download");
        }

        Self {
            server,
            payload,
            destination,
            temp_dir_guard,
        }
    }

    pub fn file_check(&self) -> FileCheck {
        FileCheck::CRC(self.payload.crc32c())
    }

    pub fn file_size(&self) -> Option<u64> {
        Some(self.payload.file.size as u64)
    }

    pub async fn wait_for_bytes(
        &self,
        minimum_bytes: u64,
    ) {
        self.server.wait_for_bytes(&self.payload.path(), minimum_bytes).await;
    }

    pub async fn wait_for_range(
        &self,
        range_start: u64,
    ) {
        self.server.wait_for_range(&self.payload.path(), range_start).await;
    }

    pub async fn release_stall(&self) {
        self.server.release_stall(&self.payload.path()).await;
    }

    pub async fn assert_downloaded(
        &self,
        state: &FileDownloadState,
    ) {
        assert!(
            state.downloaded_bytes >= self.payload.file.size as u64,
            "terminal state should report at least the file size"
        );
        assert_eq!(tokio_read(&self.destination).await.expect("downloaded file missing"), self.payload.bytes.to_vec());
        assert!(
            PathBuf::from(format!("{}.crc", self.destination.display())).exists(),
            "CRC sidecar should exist after a checked download"
        );
    }

    pub async fn expect_no_temp_artifacts(&self) {
        tokio_timeout(Duration::from_secs(2), async {
            loop {
                let artifacts = [".part", ".lock", ".resume_data"]
                    .into_iter()
                    .map(|suffix| PathBuf::from(format!("{}{}", self.destination.display(), suffix)))
                    .collect::<Vec<_>>();
                if artifacts.iter().all(|path| !path.exists()) {
                    return;
                }
                tokio_sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("temporary artifacts should be gone for {}", self.destination.display()));
    }

    pub fn part_path(&self) -> PathBuf {
        self.destination.with_extension("part")
    }

    pub fn lock_path(&self) -> PathBuf {
        PathBuf::from(format!("{}.lock", self.destination.display()))
    }

    pub fn resume_data_path(&self) -> PathBuf {
        PathBuf::from(format!("{}.resume_data", self.destination.display()))
    }
}

pub async fn wait_for_progress_bytes(
    task: &Arc<dyn FileDownloadTask>,
    progress_stream: &mut BroadcastStream<FileDownloadState>,
    minimum_bytes: u64,
) -> FileDownloadState {
    tokio_timeout(Duration::from_secs(15), async {
        loop {
            let state = task.state().await;
            if state.downloaded_bytes >= minimum_bytes {
                return state;
            }

            if let Some(Ok(state)) = progress_stream.next().await {
                if state.downloaded_bytes >= minimum_bytes {
                    return state;
                }
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {} downloaded bytes", minimum_bytes))
}

pub async fn wait_for_terminal_phase(
    task: &Arc<dyn FileDownloadTask>,
    progress_stream: &mut BroadcastStream<FileDownloadState>,
) -> FileDownloadState {
    tokio_timeout(Duration::from_secs(15), async {
        loop {
            let state = task.state().await;
            if matches!(state.phase, FileDownloadPhase::Downloaded | FileDownloadPhase::Error(_)) {
                return state;
            }

            if let Some(Ok(state)) = progress_stream.next().await {
                if matches!(state.phase, FileDownloadPhase::Downloaded | FileDownloadPhase::Error(_)) {
                    return state;
                }
            }
        }
    })
    .await
    .expect("timed out waiting for terminal phase")
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
    .unwrap_or_else(|_| panic!("timed out waiting for {:?}", phase_kind))
}

pub fn error_message(state: FileDownloadState) -> String {
    let FileDownloadPhase::Error(message) = state.phase else {
        unreachable!("expected error state")
    };
    message
}

pub fn download_manager_test_name(download_manager_type: FileDownloadManagerType) -> &'static str {
    match download_manager_type {
        FileDownloadManagerType::Apple => "apple",
        FileDownloadManagerType::Universal => "universal",
    }
}
