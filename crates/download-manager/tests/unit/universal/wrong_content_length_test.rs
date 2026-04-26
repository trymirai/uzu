use crate::common::{
    mock_download_server::{MockFile, RouteBehavior},
    scenarios::{DownloadScenario, ManagerKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_wrong_content_length_fails() {
    let scenario = DownloadScenario::new(
        ManagerKind::Universal,
        MockFile::tokenizer("download-manager/universal-wrong-length"),
        RouteBehavior::WrongContentLength,
    )
    .await;

    scenario.start_download().await;
    let message = scenario.expect_error().await;
    assert!(!message.is_empty(), "wrong content length should surface an error");
}
