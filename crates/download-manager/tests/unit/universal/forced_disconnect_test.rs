use crate::common::{
    mock_download_server::{MockFile, RouteBehavior},
    scenarios::{DownloadScenario, ManagerKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_forced_disconnect_fails_or_retries_cleanly() {
    let scenario = DownloadScenario::new(
        ManagerKind::Universal,
        MockFile::tokenizer("download-manager/universal-disconnect"),
        RouteBehavior::DisconnectAt {
            byte_offset: 8 * 1024,
        },
    )
    .await;

    scenario.start_download().await;
    let message = scenario.expect_error().await;
    assert!(!message.is_empty(), "disconnect should surface an error");
}
