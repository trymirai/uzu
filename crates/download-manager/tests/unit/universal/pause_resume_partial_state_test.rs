use crate::common::{
    mock_download_server::{MockFile, RouteBehavior},
    scenarios::{DownloadScenario, ManagerKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_pause_resume_completes_from_partial_state() {
    let scenario = DownloadScenario::new(
        ManagerKind::Universal,
        MockFile::large_tokenizer("download-manager/universal-range-resume"),
        RouteBehavior::SlowChunks {
            chunk_size: 16 * 1024,
            delay_ms: 1,
        },
    )
    .await;

    scenario.start_download().await;
    scenario.wait_for_progress_bytes(64 * 1024).await;
    let paused_state = scenario.pause_download().await;
    assert!(paused_state.downloaded_bytes > 0);

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    scenario.resume_download().await;
    scenario.expect_downloaded().await;
}
