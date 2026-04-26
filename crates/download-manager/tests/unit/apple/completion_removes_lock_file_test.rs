use crate::common::{
    mock_download_server::{MockFile, RouteBehavior},
    scenarios::{DownloadScenario, ManagerKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_completion_removes_lock_file() {
    let scenario = DownloadScenario::new(
        ManagerKind::Apple,
        MockFile::tokenizer("download-manager/apple-lock-cleanup"),
        RouteBehavior::SlowChunks {
            chunk_size: 16 * 1024,
            delay_ms: 1,
        },
    )
    .await;

    scenario.start_download().await;
    scenario.expect_downloaded().await;
    scenario.expect_no_temp_artifacts().await;
}
