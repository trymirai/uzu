use crate::common::{
    mock_download_server::RouteBehavior,
    mock_storage::{StorageFixture, StoragePhaseKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_relaunch_preserves_completed_files() {
    let fixture = StorageFixture::new(RouteBehavior::Normal).await;
    fixture.download().await;
    fixture.wait_for_phase(StoragePhaseKind::Downloaded).await;

    let storage = fixture.recreate_storage().await;
    let state = storage.state(&fixture.model.identifier).await.expect("model should survive refresh");

    assert!(matches!(state.phase, uzu::storage::types::DownloadPhase::Downloaded {}));
    assert_eq!(state.downloaded_bytes, state.total_bytes);
}
