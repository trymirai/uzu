use std::sync::Arc;

use chrono::Utc;
use download_manager::FileDownloadManagerType;
use kiban::rt::RuntimeHandle;
use mock_registry::{Behavior, MockRegistry};
use rstest::rstest;
use tokio::time::{Duration, timeout};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};
use uzu::{
    engine::Downloader,
    helpers::SharedAccess,
    storage::types::{DownloadPhase, DownloadState, Item},
};

use crate::common::{test_storage::TestStorage, tracing_setup::init_test_tracing};

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_storage_mock_registry_model_download_lifecycle(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    init_test_tracing();
    tracing::info!("starting storage mock registry lifecycle test");
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let model = registry.models.first().ok_or_else(|| std::io::Error::other("mock registry must include a model"))?;
    tracing::info!(model_identifier = %model.identifier, "loaded mock registry model");
    let test_storage =
        TestStorage::with_models_and_manager(RuntimeHandle::current(), vec![model.clone()], download_manager_type)
            .await?;
    let model_identifier = model.identifier.clone();
    let item = Arc::new(
        test_storage
            .storage
            .get(&model_identifier)
            .await
            .ok_or_else(|| format!("model not found: {model_identifier}"))?,
    );
    let mut progress = item.progress().await?;
    let total_bytes = item.state().await.total_bytes;

    tracing::info!(total_bytes, "starting first download");
    item.download().await?;
    let pre_pause_state = wait_for_item_state(&item, &mut progress, "first download reaches 25%", |state| {
        has_reached_fraction(state, 1, 4)
    })
    .await;
    assert!(
        matches!(pre_pause_state.phase, DownloadPhase::Downloading {}),
        "must pause while download is still active to exercise pause/resume; got {:?}",
        pre_pause_state.phase
    );

    tracing::info!("pausing storage item");
    item.pause().await?;
    item.reconcile().await?;
    let paused_state = wait_for_item_state(&item, &mut progress, "pause transitions to Paused", |state| {
        matches!(state.phase, DownloadPhase::Paused {})
    })
    .await;
    tracing::info!(?paused_state, "observed paused state");

    tracing::info!("resuming storage item");
    item.download().await?;
    let post_resume_state = wait_for_item_state(&item, &mut progress, "resumed download reaches 50%", |state| {
        has_reached_fraction(state, 1, 2) || matches!(state.phase, DownloadPhase::Downloaded {})
    })
    .await;
    assert!(
        matches!(post_resume_state.phase, DownloadPhase::Downloading {} | DownloadPhase::Downloaded {}),
        "resumed model must transition through Downloading and reach Downloaded; got {:?}",
        post_resume_state.phase
    );

    tracing::info!("cancelling storage item");
    item.cancel().await?;
    let cancelled_state = item.state().await;
    tracing::info!(?cancelled_state, "observed cancelled state");
    assert!(matches!(cancelled_state.phase, DownloadPhase::NotDownloaded {}));
    assert_eq!(cancelled_state.downloaded_bytes, 0);

    tracing::info!("starting final download");
    item.download().await?;
    let final_state = wait_for_item_state(&item, &mut progress, "final download completes", |state| {
        matches!(state.phase, DownloadPhase::Downloaded {})
    })
    .await;
    tracing::info!(?final_state, "observed final downloaded state");

    assert_eq!(final_state.downloaded_bytes, total_bytes);
    assert_eq!(final_state.total_bytes, total_bytes);
    for served_file in registry.files {
        let destination = item.cache_path.join(&served_file.file.name);
        assert_eq!(tokio::fs::read(&destination).await?, served_file.bytes.to_vec());
        assert!(std::path::PathBuf::from(format!("{}.crc", destination.display())).is_file());
    }

    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_storage_cancel_does_not_delete_locked_files(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    init_test_tracing();
    let registry = MockRegistry::start().await?;
    let model = registry.models.first().ok_or_else(|| std::io::Error::other("mock registry must include a model"))?;
    let test_storage =
        TestStorage::with_models_and_manager(RuntimeHandle::current(), vec![model.clone()], download_manager_type)
            .await?;
    let item = test_storage
        .storage
        .get(&model.identifier)
        .await
        .ok_or_else(|| format!("model not found: {}", model.identifier))?;
    let served_file =
        registry.files.first().ok_or_else(|| std::io::Error::other("mock registry must include files"))?;
    let destination = item.cache_path.join(&served_file.file.name);
    if let Some(parent) = destination.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(&destination, served_file.bytes.as_ref()).await?;
    tokio::fs::write(
        std::path::PathBuf::from(format!("{}.lock", destination.display())),
        serde_json::to_vec(&serde_json::json!({
            "manager_id": "foreign-manager",
            "acquired_at": Utc::now(),
            "process_id": std::process::id(),
        }))?,
    )
    .await?;

    let result = item.cancel().await;

    assert!(result.is_err());
    assert!(destination.exists());
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_downloader_progress_stream_closes_after_pause(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    init_test_tracing();
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let model = registry.models.first().ok_or_else(|| std::io::Error::other("mock registry must include a model"))?;
    let test_storage =
        TestStorage::with_models_and_manager(RuntimeHandle::current(), vec![model.clone()], download_manager_type)
            .await?;
    let downloader = Downloader::new(model.identifier.clone(), SharedAccess::new(test_storage.storage));

    downloader.resume().await?;
    let progress = downloader.progress().await?;
    timeout(Duration::from_secs(30), async {
        loop {
            let state = downloader.state().await.expect("model state must exist");
            if matches!(state.phase, DownloadPhase::Downloading {}) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await?;

    downloader.pause().await?;
    timeout(Duration::from_secs(10), async { while progress.next().await.is_some() {} }).await?;

    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_downloader_pause_updates_public_state(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    init_test_tracing();
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let model = registry.models.first().ok_or_else(|| std::io::Error::other("mock registry must include a model"))?;
    let test_storage =
        TestStorage::with_models_and_manager(RuntimeHandle::current(), vec![model.clone()], download_manager_type)
            .await?;
    let downloader = Downloader::new(model.identifier.clone(), SharedAccess::new(test_storage.storage));

    downloader.resume().await?;
    wait_for_downloader_state(&downloader, "downloader reaches Downloading", |state| {
        matches!(state.phase, DownloadPhase::Downloading {})
    })
    .await;

    downloader.pause().await?;
    let paused_state = wait_for_downloader_state(&downloader, "downloader reaches Paused", |state| {
        matches!(state.phase, DownloadPhase::Paused {})
    })
    .await;

    assert!(!paused_state.is_in_progress());
    assert!(!paused_state.can_pause());
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_downloader_progress_after_resuming_paused_model_reaches_downloaded(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    init_test_tracing();
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let model = registry.models.first().ok_or_else(|| std::io::Error::other("mock registry must include a model"))?;
    let test_storage =
        TestStorage::with_models_and_manager(RuntimeHandle::current(), vec![model.clone()], download_manager_type)
            .await?;
    let downloader = Downloader::new(model.identifier.clone(), SharedAccess::new(test_storage.storage));

    downloader.resume().await?;
    wait_for_downloader_state(&downloader, "downloader reaches Downloading before pause", |state| {
        matches!(state.phase, DownloadPhase::Downloading {})
    })
    .await;
    downloader.pause().await?;
    wait_for_downloader_state(&downloader, "downloader reaches Paused before resume", |state| {
        matches!(state.phase, DownloadPhase::Paused {})
    })
    .await;

    downloader.resume().await?;
    let progress = downloader.progress().await?;
    timeout(Duration::from_secs(30), async {
        loop {
            let state = downloader.state().await.expect("model state must exist");
            if matches!(state.phase, DownloadPhase::Downloaded {}) {
                return;
            }
            assert!(progress.next().await.is_some(), "progress stream ended before the resumed model downloaded");
        }
    })
    .await?;
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_downloader_progress_for_downloaded_model_is_empty(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    init_test_tracing();
    let registry = MockRegistry::start().await?;
    let model = registry.models.first().ok_or_else(|| std::io::Error::other("mock registry must include a model"))?;
    let test_storage =
        TestStorage::with_models_and_manager(RuntimeHandle::current(), vec![model.clone()], download_manager_type)
            .await?;
    let downloader = Downloader::new(model.identifier.clone(), SharedAccess::new(test_storage.storage));

    downloader.resume().await?;
    wait_for_downloader_state(&downloader, "downloader reaches Downloaded", |state| {
        matches!(state.phase, DownloadPhase::Downloaded {})
    })
    .await;

    let downloaded_progress = downloader.progress().await?;
    assert!(downloaded_progress.next().await.is_none());
    Ok(())
}

async fn wait_for_item_state(
    item: &Arc<Item>,
    progress: &mut BroadcastStream<DownloadState>,
    label: &'static str,
    mut is_expected_state: impl FnMut(&DownloadState) -> bool,
) -> DownloadState {
    timeout(Duration::from_secs(30), async {
        tracing::info!(label, "waiting for storage download state");
        let current_state = item.state().await;
        tracing::debug!(label, ?current_state, "observed initial storage download state");
        if is_expected_state(&current_state) {
            return current_state;
        }

        while let Some(result) = progress.next().await {
            let state = result.expect("storage progress stream must not lag");
            tracing::debug!(label, ?state, "observed storage progress state");
            if is_expected_state(&state) {
                return state;
            }
        }

        panic!("storage progress stream ended before expected state");
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for storage download state: {label}"))
}

async fn wait_for_downloader_state(
    downloader: &Downloader,
    label: &'static str,
    mut is_expected_state: impl FnMut(&DownloadState) -> bool,
) -> DownloadState {
    timeout(Duration::from_secs(30), async {
        loop {
            let state = downloader.state().await.expect("model state must exist");
            if is_expected_state(&state) {
                return state;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for downloader state: {label}"))
}

fn has_reached_fraction(
    state: &DownloadState,
    numerator: i64,
    denominator: i64,
) -> bool {
    state.total_bytes > 0 && state.downloaded_bytes >= state.total_bytes * numerator / denominator
}
