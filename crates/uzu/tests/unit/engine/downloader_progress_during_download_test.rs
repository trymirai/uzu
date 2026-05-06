use download_manager::FileDownloadManagerType;
use mock_registry::{Behavior, MockRegistry};
use rstest::rstest;
use tokio::runtime::Handle as TokioHandle;
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
async fn test_downloader_progress_during_active_download_streams_events_until_completion(
    #[case] download_manager_type: FileDownloadManagerType,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let model = registry.models.first().expect("mock registry must include a model");

    let temp_dir = tempfile::tempdir()?;
    let device = Device::new()?;
    let config = Config::new(device, Some(temp_dir.path().to_path_buf()), "test_storage".to_string())
        .with_download_manager_type(download_manager_type);
    let storage = Storage::new(TokioHandle::current(), config).await?;
    storage.refresh(vec![model.clone()]).await?;
    let storage_shared = SharedAccess::new(storage);

    let downloader = Downloader::new(model.identifier.clone(), storage_shared.clone());
    let initial_phase = downloader.state().await.expect("state present").phase;
    assert!(
        !matches!(initial_phase, DownloadPhase::Downloaded {}),
        "precondition: model must not be Downloaded yet, got {:?}",
        initial_phase
    );

    storage_shared.lock().await.download(&model.identifier).await?;
    let stream = downloader.progress().await?;

    let mut update_count = 0;
    while let Some(_update) = stream.next().await {
        update_count += 1;
    }
    assert!(update_count > 0, "expected at least one progress update during an active download");

    let final_phase = downloader.state().await.expect("state present").phase;
    assert!(
        matches!(final_phase, DownloadPhase::Downloaded {}),
        "expected final phase to be Downloaded, got {:?}",
        final_phase
    );

    Ok(())
}
