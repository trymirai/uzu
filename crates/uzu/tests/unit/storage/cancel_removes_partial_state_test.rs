use crate::common::{
    mock_download_server::RouteBehavior,
    mock_storage::{StorageFixture, StoragePhaseKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_cancel_removes_partial_model_state() {
    let fixture = StorageFixture::with_tokenizer_behavior(RouteBehavior::StallAt {
        byte_offset: 64 * 1024,
    })
    .await;

    fixture.download().await;
    let tokenizer = fixture.registry_fixture.payload("tokenizer.json");
    fixture.server.wait_for_bytes(&tokenizer.path(), 64 * 1024).await;
    fixture.delete().await;
    let state = fixture.wait_for_phase(StoragePhaseKind::NotDownloaded).await;

    assert_eq!(state.downloaded_bytes, 0);
}
