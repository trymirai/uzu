use crate::common::{
    mock_download_server::RouteBehavior,
    mock_storage::StorageFixture,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_relaunch_preserves_partial_files_as_paused() {
    let fixture = StorageFixture::with_tokenizer_behavior(RouteBehavior::StallAt {
        byte_offset: 64 * 1024,
    })
    .await;
    fixture.download().await;
    let tokenizer = fixture.registry_fixture.payload("tokenizer.json");
    fixture.server.wait_for_bytes(&tokenizer.path(), 64 * 1024).await;
    fixture.pause().await;

    let storage = fixture.recreate_storage().await;
    let state = storage.state(&fixture.model.identifier).await.expect("model should survive refresh");

    assert!(matches!(state.phase, uzu::storage::types::DownloadPhase::Paused {}));
    assert!(state.downloaded_bytes > 0);
}
