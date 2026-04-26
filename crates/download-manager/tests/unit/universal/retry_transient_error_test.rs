use crate::common::{
    mock_download_server::{MockFile, RouteBehavior},
    scenarios::{DownloadScenario, ManagerKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_retries_transient_server_error() {
    let scenario = DownloadScenario::new(
        ManagerKind::Universal,
        MockFile::tokenizer("download-manager/universal-retry"),
        RouteBehavior::RetryThenOk {
            failures: 1,
            status: 500,
        },
    )
    .await;

    scenario.start_download().await;
    scenario.expect_downloaded().await;

    let statuses = scenario.server.records_snapshot().await.into_iter().map(|record| record.status).collect::<Vec<_>>();
    assert!(statuses.contains(&500), "mock server should have produced a retryable failure");
    assert!(statuses.contains(&200) || statuses.contains(&206), "download should retry and complete");
}
