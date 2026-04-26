use crate::common::{
    mock_download_server::RouteBehavior,
    mock_storage::{StorageFixture, StoragePhaseKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_broadcasts_progress_and_downloaded_phase() {
    let fixture = StorageFixture::new(RouteBehavior::SlowChunks {
        chunk_size: 16 * 1024,
        delay_ms: 1,
    })
    .await;

    fixture.download().await;
    let downloading = fixture.wait_for_broadcast_phase(StoragePhaseKind::Downloading).await;
    let downloaded = fixture.wait_for_broadcast_phase(StoragePhaseKind::Downloaded).await;

    assert!(downloading.downloaded_bytes <= downloading.total_bytes);
    assert_eq!(downloaded.downloaded_bytes, downloaded.total_bytes);
}
