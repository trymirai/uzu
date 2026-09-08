use std::{path::Path, time::Duration};

use download_manager::{DestinationLock, DownloadState, DownloadTask, DownloadTaskRequest, LockOwner};
use kiban::stream::BoxStream;
use mock_registry::MockRegistry;
use tokio::time::timeout;
use tokio_stream::StreamExt;
use uuid::Uuid;

pub fn file_request(
    source_url: &str,
    destination: &Path,
    expected_crc32c: Option<String>,
    expected_bytes: Option<u64>,
) -> DownloadTaskRequest {
    DownloadTaskRequest::file()
        .destination(destination)
        .source_url(source_url)
        .maybe_expected_crc32c(expected_crc32c)
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
                .expected_crc32c(served.crc32c()?)
                .expected_bytes(served.file.size as u64)
                .build(),
        );
    }
    Ok(DownloadTaskRequest::group().destination(directory).subrequests(files).build())
}

pub async fn wait_for_state(
    task: &DownloadTask,
    progress: &mut BoxStream<'static, DownloadState>,
    mut is_expected: impl FnMut(&DownloadState) -> bool,
) -> DownloadState {
    timeout(Duration::from_secs(30), async {
        let state = task.state();
        if is_expected(&state) {
            return state;
        }
        while let Some(state) = progress.next().await {
            if is_expected(&state) {
                return state;
            }
        }
        panic!("progress stream ended before the expected state");
    })
    .await
    .expect("timed out waiting for download state")
}

pub async fn foreign_lock(destination: &Path) -> DestinationLock {
    let owner = LockOwner {
        manager_id: "foreign-manager".to_string(),
        instance_id: Uuid::new_v4(),
    };
    DestinationLock::acquire(destination, &owner).await.expect("foreign lock")
}
