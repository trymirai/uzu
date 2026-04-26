use crate::common::{
    mock_download_server::{MockFile, RouteBehavior},
    scenarios::{DownloadScenario, ManagerKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_resume_consumes_resume_data_file() {
    let scenario = DownloadScenario::new(
        ManagerKind::Apple,
        MockFile::large_tokenizer("download-manager/apple-resume-consume"),
        RouteBehavior::SlowChunks {
            chunk_size: 16 * 1024,
            delay_ms: 1,
        },
    )
    .await;

    scenario.start_download().await;
    scenario.wait_for_progress_bytes(64 * 1024).await;
    scenario.pause_download().await;
    assert!(scenario.resume_data_path().exists(), "pause should persist resume data");

    scenario.resume_download().await;
    assert!(!scenario.resume_data_path().exists(), "resume should consume persisted resume data");
    scenario.expect_downloaded().await;
}
