use download_manager::FileDownloadManagerType;
use mock_registry::Behavior;
use rstest::rstest;
use uzu::storage::types::DownloadPhase;

use crate::common::test_engine_fixture::TestEngineFixture;

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn progress_during_active_download_streams_through_completion(
    #[case] download_manager_type: FileDownloadManagerType,
) -> Result<(), Box<dyn std::error::Error>> {
    let fixture = TestEngineFixture::start_with(download_manager_type, Behavior::THROTTLED).await?;

    fixture.storage.lock().await.download(&fixture.model.identifier).await?;
    let stream = fixture.downloader.progress().await?;

    let mut updates: Vec<(i64, i64)> = Vec::new();
    while let Some(update) = stream.next().await {
        updates.push((update.bytes_downloaded, update.bytes_total));
    }

    let final_phase = fixture.downloader.state().await.unwrap().phase;
    assert!(matches!(final_phase, DownloadPhase::Downloaded {}), "expected Downloaded, got {:?}", final_phase);
    assert!(
        updates.iter().any(|(downloaded, total)| *total > 0 && downloaded < total),
        "expected an in-progress update; got {:?}",
        updates,
    );
    assert!(
        updates.iter().any(|(downloaded, total)| *total > 0 && downloaded == total),
        "expected a final-byte update; got {:?}",
        updates,
    );

    Ok(())
}
