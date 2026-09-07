#![cfg(not(target_family = "wasm"))]

mod common;

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use chrono::Utc;
use download_manager::DownloadManagerType;
use kiban::rt::RuntimeHandle;
use mock_registry::{Behavior, MockRegistry};
use rstest::rstest;
use tokio::time::timeout;
use tokio_stream::{StreamExt, wrappers::BroadcastStream};
use uzu::{
    engine::Downloader,
    helpers::SharedAccess,
    storage::{DownloadPhase, DownloadState, Storage},
    types::model::ModelIdentifier,
};

use crate::common::test_storage::TestStorage;

#[rstest]
#[case::universal(DownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::native(DownloadManagerType::Native))]
#[tokio::test(flavor = "multi_thread")]
async fn model_lifecycle(#[case] kind: DownloadManagerType) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let model = registry.models.first().ok_or("mock registry must include a model")?;
    let test_storage =
        TestStorage::with_models_and_manager(RuntimeHandle::current(), vec![model.clone()], kind).await?;
    let storage = &test_storage.storage;
    let identifier = model.identifier.clone();
    let cache_path = storage.cache_model_path(model).ok_or("model must have a cache path")?;
    let mut events = storage.subscribe();
    let total_bytes = storage.state(&identifier).await.ok_or("model must have a state")?.total_bytes;

    storage.download(&identifier).await?;
    let downloading = wait_for(storage, &identifier, &mut events, |state| {
        state.total_bytes > 0 && state.downloaded_bytes >= state.total_bytes / 4
    })
    .await;
    assert!(matches!(downloading.phase, DownloadPhase::Downloading {}), "got {:?}", downloading.phase);

    storage.pause(&identifier).await?;
    let paused =
        wait_for(storage, &identifier, &mut events, |state| matches!(state.phase, DownloadPhase::Paused {})).await;
    assert!(paused.downloaded_bytes >= total_bytes / 4);

    storage.download(&identifier).await?;
    let downloaded =
        wait_for(storage, &identifier, &mut events, |state| matches!(state.phase, DownloadPhase::Downloaded {})).await;
    assert_eq!(downloaded.downloaded_bytes, total_bytes);
    for served in registry.files.iter() {
        let destination = cache_path.join(&served.file.name);
        assert_eq!(tokio::fs::read(&destination).await?, served.bytes.to_vec());
        assert!(crc_path(&destination).is_file());
    }

    storage.delete(&identifier).await?;
    let deleted = storage.state(&identifier).await.ok_or("model must have a state")?;
    assert!(matches!(deleted.phase, DownloadPhase::NotDownloaded {}));
    assert_eq!(deleted.downloaded_bytes, 0);
    for served in registry.files.iter() {
        let destination = cache_path.join(&served.file.name);
        assert!(!destination.exists());
        assert!(!crc_path(&destination).exists());
    }

    let served = registry.files.first().ok_or("mock registry must include files")?;
    let destination = cache_path.join(&served.file.name);
    tokio::fs::create_dir_all(&cache_path).await?;
    tokio::fs::write(&destination, served.bytes.as_ref()).await?;
    tokio::fs::write(
        PathBuf::from(format!("{}.lock", destination.display())),
        serde_json::to_vec(&serde_json::json!({
            "manager_id": "foreign-manager",
            "acquired_at": Utc::now(),
            "process_id": std::process::id(),
        }))?,
    )
    .await?;
    assert!(storage.delete(&identifier).await.is_err());
    assert!(destination.exists());
    Ok(())
}

#[rstest]
#[case::universal(DownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::native(DownloadManagerType::Native))]
#[tokio::test(flavor = "multi_thread")]
async fn downloader_streams(#[case] kind: DownloadManagerType) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let model = registry.models.first().ok_or("mock registry must include a model")?;
    let test_storage =
        TestStorage::with_models_and_manager(RuntimeHandle::current(), vec![model.clone()], kind).await?;
    let downloader = Downloader::new(model.identifier.clone(), SharedAccess::new(test_storage.storage));

    downloader.resume().await?;
    let progress = downloader.progress().await?;
    wait_for_downloader(&downloader, |state| matches!(state.phase, DownloadPhase::Downloading {})).await;
    downloader.pause().await?;
    timeout(Duration::from_secs(10), async { while progress.next().await.is_some() {} }).await?;
    let paused = wait_for_downloader(&downloader, |state| matches!(state.phase, DownloadPhase::Paused {})).await;
    assert!(!paused.is_in_progress());
    assert!(!paused.can_pause());

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

    assert!(downloader.progress().await?.next().await.is_none());
    Ok(())
}

async fn wait_for(
    storage: &Storage,
    identifier: &ModelIdentifier,
    events: &mut BroadcastStream<(ModelIdentifier, DownloadState)>,
    mut is_expected: impl FnMut(&DownloadState) -> bool,
) -> DownloadState {
    timeout(Duration::from_secs(30), async {
        let state = storage.state(identifier).await.expect("model state must exist");
        if is_expected(&state) {
            return state;
        }
        while let Some(result) = events.next().await {
            let (event_identifier, state) = result.expect("storage event stream must not lag");
            if &event_identifier == identifier && is_expected(&state) {
                return state;
            }
        }
        panic!("storage event stream ended before the expected state");
    })
    .await
    .expect("timed out waiting for storage state")
}

async fn wait_for_downloader(
    downloader: &Downloader,
    mut is_expected: impl FnMut(&DownloadState) -> bool,
) -> DownloadState {
    timeout(Duration::from_secs(30), async {
        loop {
            let state = downloader.state().await.expect("model state must exist");
            if is_expected(&state) {
                return state;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("timed out waiting for downloader state")
}

fn crc_path(destination: &Path) -> PathBuf {
    PathBuf::from(format!("{}.crc", destination.display()))
}
