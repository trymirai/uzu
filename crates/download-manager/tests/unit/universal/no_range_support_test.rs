use crate::common::{
    mock_download_server::{MockFile, RouteBehavior},
    scenarios::{DownloadScenario, ManagerKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_no_range_support_still_downloads_fresh_file() {
    let scenario = DownloadScenario::new(
        ManagerKind::Universal,
        MockFile::tokenizer("download-manager/universal-no-range"),
        RouteBehavior::NoRangeSupport,
    )
    .await;

    scenario.start_download().await;
    scenario.expect_downloaded().await;
}
