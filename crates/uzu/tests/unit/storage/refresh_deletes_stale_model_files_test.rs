use mock_registry::MockRegistry;
use tokio::runtime::Handle as TokioHandle;
use uzu::storage::types::DownloadPhase;

use crate::common::{test_storage::TestStorage, tracing_setup::init_test_tracing};

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_refresh_preserves_files_for_models_dropped_from_registry()
-> Result<(), Box<dyn std::error::Error>> {
    init_test_tracing();
    let registry = MockRegistry::start().await?;
    let model = registry.models.first().ok_or_else(|| std::io::Error::other("mock registry must include a model"))?;
    let test_storage = TestStorage::with_models(TokioHandle::current(), vec![model.clone()]).await?;

    let item = test_storage.storage.get(&model.identifier).await.ok_or_else(|| format!("missing item"))?;
    let cache_path = item.cache_path.clone();

    item.download().await?;
    let mut progress = item.progress().await?;
    use tokio_stream::StreamExt;
    while let Some(state) = progress.next().await {
        let state = state.expect("progress stream must not lag");
        if matches!(state.phase, DownloadPhase::Downloaded {}) {
            break;
        }
    }
    drop(item);

    for served in registry.files.iter() {
        let on_disk = cache_path.join(&served.file.name);
        assert!(on_disk.exists(), "precondition: model file present after download");
    }

    test_storage.storage.refresh(Vec::new()).await?;

    for served in registry.files.iter() {
        let on_disk = cache_path.join(&served.file.name);
        assert!(
            on_disk.exists(),
            "Storage::refresh removed downloaded file {} when its model disappeared from the registry. \
             A registry update should not destroy user-owned bytes already on disk.",
            on_disk.display()
        );
    }

    Ok(())
}
