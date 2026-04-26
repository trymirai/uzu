use crate::common::{
    mock_download_server::{MockFile, RouteBehavior},
    scenarios::{DownloadScenario, ManagerKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_broadcast_reports_terminal_phase() {
    let scenario = DownloadScenario::new(
        ManagerKind::Universal,
        MockFile::tokenizer("download-manager/universal-broadcast"),
        RouteBehavior::Normal,
    )
    .await;
    let mut progress = scenario.task.progress().await.expect("progress stream should open");

    scenario.start_download().await;
    scenario.expect_downloaded().await;

    let observed_downloaded = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while let Some(Ok(state)) = tokio_stream::StreamExt::next(&mut progress).await {
            if matches!(state.phase, download_manager::FileDownloadPhase::Downloaded) {
                return true;
            }
        }
        false
    })
    .await
    .unwrap_or(false);
    assert!(observed_downloaded, "progress stream should publish Downloaded");
}
