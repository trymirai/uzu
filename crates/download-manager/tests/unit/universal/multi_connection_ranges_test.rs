use crate::common::{
    mock_download_server::{MockFile, RouteBehavior},
    scenarios::{DownloadScenario, ManagerKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_multi_connection_requests_byte_ranges() {
    let scenario = DownloadScenario::new(
        ManagerKind::Universal,
        MockFile::large_tokenizer("download-manager/universal-range-split"),
        RouteBehavior::Normal,
    )
    .await;

    scenario.start_download().await;
    scenario.expect_downloaded().await;

    let range_requests = scenario
        .server
        .records_snapshot()
        .await
        .into_iter()
        .filter(|record| record.range.is_some())
        .collect::<Vec<_>>();
    assert!(!range_requests.is_empty(), "universal downloader should request byte ranges");
}
