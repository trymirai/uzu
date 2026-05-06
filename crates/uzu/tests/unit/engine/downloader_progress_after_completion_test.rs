use download_manager::FileDownloadManagerType;
use rstest::rstest;
use tokio_stream::StreamExt;
use uzu::storage::types::DownloadPhase;

use crate::common::test_engine_fixture::TestEngineFixture;

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn progress_after_completion_returns_empty_stream_not_error(
    #[case] download_manager_type: FileDownloadManagerType,
) -> Result<(), Box<dyn std::error::Error>> {
    let fixture = TestEngineFixture::start(download_manager_type).await?;

    let item = fixture.storage.lock().await.get(&fixture.model.identifier).await.unwrap();
    let mut item_progress = item.progress().await?;
    item.download().await?;
    while let Some(state) = item_progress.next().await {
        let state = state?;
        if matches!(state.phase, DownloadPhase::Downloaded {}) {
            break;
        }
    }
    assert!(matches!(item.state().await.phase, DownloadPhase::Downloaded {}));

    let stream = fixture.downloader.progress().await?;
    assert!(stream.next().await.is_none(), "stream for an already-downloaded model must yield no updates");

    Ok(())
}
