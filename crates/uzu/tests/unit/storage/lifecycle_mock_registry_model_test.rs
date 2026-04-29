use std::sync::Arc;

use mock_registry::{Behavior, MockRegistry};
use tokio::runtime::Handle as TokioHandle;
use tokio::time::{Duration, timeout};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};
use uzu::storage::types::{DownloadPhase, DownloadState, Item};

use crate::common::{test_storage::TestStorage, tracing_setup::init_test_tracing};

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_mock_registry_model_download_lifecycle() -> Result<(), Box<dyn std::error::Error>> {
    init_test_tracing();
    tracing::info!("starting storage mock registry lifecycle test");
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let model = registry.models.first().ok_or_else(|| std::io::Error::other("mock registry must include a model"))?;
    tracing::info!(model_identifier = %model.identifier, "loaded mock registry model");
    let test_storage = TestStorage::with_models(TokioHandle::current(), vec![model.clone()]).await?;
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
    wait_for_item_state(&item, &mut progress, "first download reaches 25%", |state| {
        has_reached_fraction(state, 1, 4) || matches!(state.phase, DownloadPhase::Downloaded {})
    })
    .await;

    tracing::info!("pausing storage item");
    item.pause().await?;
    item.reconcile().await?;
    let paused_state = item.state().await;
    tracing::info!(?paused_state, "observed paused state");
    assert!(matches!(paused_state.phase, DownloadPhase::Paused {} | DownloadPhase::Downloaded {}));

    tracing::info!("resuming storage item");
    item.download().await?;
    wait_for_item_state(&item, &mut progress, "resumed download reaches 50%", |state| {
        has_reached_fraction(state, 1, 2) || matches!(state.phase, DownloadPhase::Downloaded {})
    })
    .await;

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

fn has_reached_fraction(
    state: &DownloadState,
    numerator: i64,
    denominator: i64,
) -> bool {
    state.total_bytes > 0 && state.downloaded_bytes >= state.total_bytes * numerator / denominator
}
