use std::sync::Arc;

use download_manager::FileDownloadManagerType;
use mock_registry::MockRegistry;
use rstest::rstest;
use tokio::runtime::Handle as TokioHandle;
use tokio_stream::StreamExt;
use uzu::{
    device::Device,
    engine::Downloader,
    helpers::SharedAccess,
    storage::{Config, Storage, types::DownloadPhase},
};

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_downloader_progress_after_completion_returns_empty_stream_not_error(
    #[case] download_manager_type: FileDownloadManagerType,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let model = registry.models.first().expect("mock registry must include a model");

    let temp_dir = tempfile::tempdir()?;
    let device = Device::new()?;
    let config = Config::new(device, Some(temp_dir.path().to_path_buf()), "test_storage".to_string())
        .with_download_manager_type(download_manager_type);
    let storage = Storage::new(TokioHandle::current(), config).await?;
    storage.refresh(vec![model.clone()]).await?;

    let item = Arc::new(storage.get(&model.identifier).await.expect("item present after refresh"));
    let mut item_progress = item.progress().await?;
    item.download().await?;
    while let Some(state) = item_progress.next().await {
        let state = state.expect("progress stream must not lag");
        if matches!(state.phase, DownloadPhase::Downloaded {}) {
            break;
        }
    }
    assert!(matches!(item.state().await.phase, DownloadPhase::Downloaded {}));
    drop(item);
    drop(item_progress);

    let storage_shared = SharedAccess::new(storage);
    let downloader = Downloader::new(model.identifier.clone(), storage_shared);
    let progress_result = downloader.progress().await;
    assert!(
        progress_result.is_ok(),
        "Downloader::progress() must return an empty stream for an already-downloaded model, got error: {:?}",
        progress_result.err()
    );

    let stream = progress_result.unwrap();
    let next = stream.next().await;
    assert!(next.is_none(), "stream for an already-downloaded model must yield no further updates, got {:?}", next);

    Ok(())
}
