use crate::common::{
    mock_download_server::RouteBehavior,
    mock_storage::{StorageFixture, StoragePhaseKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_downloads_mock_model_files() {
    let fixture = StorageFixture::new(RouteBehavior::Normal).await;

    fixture.download().await;
    let state = fixture.wait_for_phase(StoragePhaseKind::Downloaded).await;

    assert_eq!(state.downloaded_bytes, state.total_bytes);
    fixture.assert_files_match_server().await;
}
