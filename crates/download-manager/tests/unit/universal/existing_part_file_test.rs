use crate::common::{
    mock_download_server::{MockFile, RouteBehavior},
    scenarios::{DownloadScenario, ManagerKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_existing_part_file_resumes_from_partial_size() {
    let file = MockFile::large_tokenizer("download-manager/universal-existing-part");
    let scenario = DownloadScenario::new_with_existing_part(
        ManagerKind::Universal,
        file.clone(),
        RouteBehavior::Normal,
        Some(file.bytes[..64 * 1024].to_vec()),
    )
    .await;

    let state = scenario.current_state().await;
    assert!(matches!(state.phase, download_manager::FileDownloadPhase::Paused));
    assert_eq!(state.downloaded_bytes, 64 * 1024);
}
