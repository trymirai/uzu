use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use chrono::Utc;
use download_manager::{DownloadState, DownloadTask, DownloadTaskRequest, FileCheck};
use mock_registry::MockRegistry;
use tokio::time::timeout;
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

pub fn file_request(
    source_url: &str,
    destination: &Path,
    file_check: FileCheck,
    expected_bytes: Option<u64>,
) -> DownloadTaskRequest {
    DownloadTaskRequest::file()
        .destination(destination)
        .source_url(source_url)
        .file_check(file_check)
        .maybe_expected_bytes(expected_bytes)
        .build()
}

pub fn model_request(
    registry: &MockRegistry,
    directory: &Path,
) -> Result<DownloadTaskRequest, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    for served in registry.files.iter() {
        files.push(
            DownloadTaskRequest::file()
                .destination(&served.file.name)
                .source_url(&served.file.url)
                .file_check(FileCheck::CRC(served.crc32c()?))
                .expected_bytes(served.file.size as u64)
                .build(),
        );
    }
    Ok(DownloadTaskRequest::group().destination(directory).subrequests(files).build())
}

pub async fn wait_for_state(
    task: &DownloadTask,
    progress: &mut BroadcastStream<DownloadState>,
    mut is_expected: impl FnMut(&DownloadState) -> bool,
) -> DownloadState {
    timeout(Duration::from_secs(30), async {
        let state = task.state();
        if is_expected(&state) {
            return state;
        }
        while let Some(result) = progress.next().await {
            let state = result.unwrap_or_else(|_| task.state());
            if is_expected(&state) {
                return state;
            }
        }
        panic!("progress stream ended before the expected state");
    })
    .await
    .expect("timed out waiting for download state")
}

pub fn crc_path(destination: &Path) -> PathBuf {
    PathBuf::from(format!("{}.crc", destination.display()))
}

pub fn lock_path(destination: &Path) -> PathBuf {
    PathBuf::from(format!("{}.lock", destination.display()))
}

pub fn foreign_lock() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "manager_id": "foreign-manager",
        "acquired_at": Utc::now(),
        "process_id": std::process::id(),
    }))
    .expect("lock json")
}
