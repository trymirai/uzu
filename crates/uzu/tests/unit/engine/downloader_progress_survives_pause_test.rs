use std::time::Duration;

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
async fn test_downloader_progress_stream_survives_pause_and_continues_to_downloaded(
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

    storage_shared.lock().await.download(&model.identifier).await?;
    let stream = downloader.progress().await?;

    let initial_phase_at_pause = loop {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let phase = downloader.state().await.expect("state present").phase;
        if matches!(phase, DownloadPhase::Downloading {}) {
            break phase;
        }
        if matches!(phase, DownloadPhase::Downloaded {}) {
            return Err("download completed before pause could fire — test setup too fast".into());
        }
    };
    assert!(matches!(initial_phase_at_pause, DownloadPhase::Downloading {}));
    storage_shared.lock().await.pause(&model.identifier).await?;
    storage_shared.lock().await.download(&model.identifier).await?;

    let mut update_count = 0;
    let mut saw_downloaded = false;
    while let Some(update) = stream.next().await {
        update_count += 1;
        if update.bytes_total > 0 && update.bytes_downloaded == update.bytes_total {
            saw_downloaded = true;
        }
    }

    let final_phase = downloader.state().await.expect("state present").phase;
    assert!(matches!(final_phase, DownloadPhase::Downloaded {}), "model must reach Downloaded, got {:?}", final_phase);
    assert!(
        saw_downloaded,
        "the progress stream must surface the final Downloaded update; got {} updates and never saw bytes_downloaded == bytes_total",
        update_count
    );

    Ok(())
}
