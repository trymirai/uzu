use crate::common::{
    mock_download_server::{MockFile, RouteBehavior},
    scenarios::{DownloadScenario, ManagerKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_corrupt_body_fails_crc() {
    let scenario = DownloadScenario::new(
        ManagerKind::Universal,
        MockFile::tokenizer("download-manager/universal-corrupt"),
        RouteBehavior::CorruptBody,
    )
    .await;

    scenario.start_download().await;
    let message = scenario.expect_error().await;
    assert!(message.contains("CRC") || message.contains("checksum"), "unexpected error: {message}");
}
