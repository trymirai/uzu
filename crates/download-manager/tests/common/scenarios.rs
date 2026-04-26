#![allow(dead_code)]

use std::{path::PathBuf, sync::Arc, time::Duration};

use download_manager::{
    FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase, FileDownloadState, FileDownloadTask,
    create_download_manager,
};
use tokio::sync::Mutex;
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

use crate::common::mock_download_server::{MockDownloadServer, MockFile, RouteBehavior};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ManagerKind {
    Apple,
    Universal,
}

impl ManagerKind {
    pub fn download_manager_type(self) -> FileDownloadManagerType {
        match self {
            Self::Apple => FileDownloadManagerType::Apple,
            Self::Universal => FileDownloadManagerType::Universal,
        }
    }

    pub fn test_name_prefix(self) -> &'static str {
        match self {
            Self::Apple => "apple",
            Self::Universal => "universal",
        }
    }
}

pub struct DownloadScenario {
    pub manager_kind: ManagerKind,
    pub server: MockDownloadServer,
    pub file: MockFile,
    pub destination: PathBuf,
    pub task: Arc<dyn FileDownloadTask>,
    pub manager: Box<dyn FileDownloadManager>,
    progress_stream: Mutex<BroadcastStream<FileDownloadState>>,
    temp_dir_guard: tempfile::TempDir,
}

impl DownloadScenario {
    pub async fn new(
        manager_kind: ManagerKind,
        file: MockFile,
        behavior: RouteBehavior,
    ) -> Self {
        Self::new_with_existing_part(manager_kind, file, behavior, None).await
    }

    pub async fn new_with_existing_part(
        manager_kind: ManagerKind,
        file: MockFile,
        behavior: RouteBehavior,
        existing_part: Option<Vec<u8>>,
    ) -> Self {
        let server = MockDownloadServer::start().await;
        server.serve_file(file.clone(), behavior).await;

        let temp_dir_guard = tempfile::tempdir().expect("failed to create test temp dir");
        let destination = temp_dir_guard.path().join(&file.name);
        if let Some(existing_part) = existing_part {
            tokio::fs::write(destination.with_extension("part"), existing_part)
                .await
                .expect("failed to seed partial download");
        }
        let manager_name = format!("{}_{}", manager_kind.test_name_prefix(), uuid::Uuid::new_v4());
        let manager = create_download_manager(
            manager_kind.download_manager_type(),
            Some(manager_name),
            tokio::runtime::Handle::current(),
        )
        .await
        .expect("failed to create download manager");
        let source_url = server.url_for_file(&file);
        let task = manager
            .file_download_task(&source_url, &destination, FileCheck::CRC(file.crc32c.clone()), Some(file.size))
            .await
            .expect("failed to create file download task");
        let progress_stream = Mutex::new(task.progress().await.expect("failed to open progress stream"));

        Self {
            manager_kind,
            server,
            file,
            destination,
            task,
            manager,
            progress_stream,
            temp_dir_guard,
        }
    }

    pub async fn start_download(&self) {
        self.task.download().await.expect("failed to start download");
    }

    pub async fn pause_download(&self) -> FileDownloadState {
        self.task.pause().await.expect("failed to pause download");
        self.wait_for_phase_kind(PhaseKind::Paused).await
    }

    pub async fn resume_download(&self) {
        self.task.download().await.expect("failed to resume download");
    }

    pub async fn cancel_download(&self) -> FileDownloadState {
        self.task.cancel().await.expect("failed to cancel download");
        self.wait_for_phase_kind(PhaseKind::NotDownloaded).await
    }

    pub async fn wait_for_bytes(
        &self,
        minimum_bytes: u64,
    ) {
        self.server.wait_for_bytes(&self.file.path, minimum_bytes).await;
    }

    pub async fn wait_for_progress_bytes(
        &self,
        minimum_bytes: u64,
    ) -> FileDownloadState {
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                let state = self.task.state().await;
                if state.downloaded_bytes >= minimum_bytes {
                    return state;
                }

                let mut progress_stream = self.progress_stream.lock().await;
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

    pub async fn wait_for_range(
        &self,
        range_start: u64,
    ) {
        self.server.wait_for_range(&self.file.path, range_start).await;
    }

    pub async fn release_stall(&self) {
        self.server.release_stall(&self.file.path).await;
    }

    pub async fn expect_downloaded(&self) {
        let state = self.wait_for_phase_kind(PhaseKind::Downloaded).await;
        assert_eq!(state.downloaded_bytes, self.file.size);
        assert_eq!(
            tokio::fs::read(&self.destination).await.expect("downloaded file missing"),
            self.file.bytes.to_vec()
        );
        assert!(
            PathBuf::from(format!("{}.crc", self.destination.display())).exists(),
            "CRC sidecar should exist after a checked download"
        );
    }

    pub async fn expect_error(&self) -> String {
        let state = self.wait_for_phase_kind(PhaseKind::Error).await;
        let FileDownloadPhase::Error(message) = state.phase else {
            unreachable!("wait_for_phase_kind returned non-error state")
        };
        message
    }

    pub async fn expect_no_temp_artifacts(&self) {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let artifacts = [".part", ".lock", ".resume_data"]
                    .into_iter()
                    .map(|suffix| PathBuf::from(format!("{}{}", self.destination.display(), suffix)))
                    .collect::<Vec<_>>();
                if artifacts.iter().all(|path| !path.exists()) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
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

    pub async fn current_state(&self) -> FileDownloadState {
        self.task.state().await
    }

    async fn wait_for_phase_kind(
        &self,
        phase_kind: PhaseKind,
    ) -> FileDownloadState {
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                let state = self.task.state().await;
                if phase_kind.matches(&state.phase) {
                    return state;
                }

                let mut progress_stream = self.progress_stream.lock().await;
                if let Some(Ok(state)) = progress_stream.next().await {
                    if phase_kind.matches(&state.phase) {
                        return state;
                    }
                }
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {:?} on {:?}", phase_kind, self.destination))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PhaseKind {
    NotDownloaded,
    Paused,
    Downloaded,
    Error,
}

impl PhaseKind {
    fn matches(
        self,
        phase: &FileDownloadPhase,
    ) -> bool {
        match self {
            Self::NotDownloaded => matches!(phase, FileDownloadPhase::NotDownloaded),
            Self::Paused => matches!(phase, FileDownloadPhase::Paused),
            Self::Downloaded => matches!(phase, FileDownloadPhase::Downloaded),
            Self::Error => matches!(phase, FileDownloadPhase::Error(_)),
        }
    }
}

pub async fn run_fresh_download_scenario(manager_kind: ManagerKind) {
    let scenario =
        DownloadScenario::new(manager_kind, MockFile::tokenizer("download-manager/fresh"), RouteBehavior::Normal).await;

    scenario.start_download().await;
    scenario.expect_downloaded().await;
    scenario.expect_no_temp_artifacts().await;
}

pub async fn run_cancel_redownload_scenario(manager_kind: ManagerKind) {
    let scenario = DownloadScenario::new(
        manager_kind,
        MockFile::large_tokenizer("download-manager/cancel"),
        RouteBehavior::SlowChunks {
            chunk_size: 16 * 1024,
            delay_ms: 1,
        },
    )
    .await;

    scenario.start_download().await;
    scenario.wait_for_progress_bytes(64 * 1024).await;
    scenario.cancel_download().await;
    tokio::time::sleep(Duration::from_millis(200)).await;

    if manager_kind == ManagerKind::Universal {
        return;
    }

    scenario.resume_download().await;
    scenario.expect_downloaded().await;
}

pub async fn run_pause_resume_scenario(manager_kind: ManagerKind) {
    let scenario = DownloadScenario::new(
        manager_kind,
        MockFile::large_tokenizer("download-manager/pause-resume"),
        RouteBehavior::SlowChunks {
            chunk_size: 16 * 1024,
            delay_ms: 1,
        },
    )
    .await;

    scenario.start_download().await;
    scenario.wait_for_progress_bytes(64 * 1024).await;
    let paused_state = scenario.pause_download().await;
    assert!(paused_state.downloaded_bytes > 0, "pause should preserve positive progress");
    tokio::time::sleep(Duration::from_millis(200)).await;

    scenario.release_stall().await;
    scenario.resume_download().await;
    scenario.expect_downloaded().await;
}
