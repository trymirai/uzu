use std::time::Duration;

use mock_registry::MockRegistry;
use tokio::runtime::Handle as TokioHandle;
use tokio::time::timeout;
use tokio_stream::StreamExt;
use uzu::storage::types::DownloadPhase;

use crate::common::test_storage::TestStorage;

#[tokio::test(flavor = "multi_thread")]
async fn test_storage_downloads_mock_registry_model_files() -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let model = registry.models.first().ok_or_else(|| std::io::Error::other("mock registry must include a model"))?;
    let test_storage = TestStorage::with_models(TokioHandle::current(), vec![model.clone()]).await?;
    let model_identifier = model.identifier.clone();
    let item = test_storage
        .storage
        .get(&model_identifier)
        .await
        .ok_or_else(|| format!("model not found: {model_identifier}"))?;
    let mut progress = item.progress().await?;

    item.download().await?;

    let final_state = timeout(Duration::from_secs(60), async {
        loop {
            let state = item.state().await;
            if matches!(state.phase, DownloadPhase::Downloaded {}) {
                return state;
            }

            if let Some(Ok(state)) = progress.next().await {
                if matches!(state.phase, DownloadPhase::Downloaded {}) {
                    return state;
                }
            }
        }
    })
    .await?;

    let expected_bytes = registry.files.iter().map(|served_file| served_file.file.size).sum::<i64>();
    assert_eq!(final_state.downloaded_bytes, expected_bytes);
    assert_eq!(final_state.total_bytes, expected_bytes);

    for served_file in registry.files {
        let destination = item.cache_path.join(&served_file.file.name);
        assert_eq!(tokio::fs::read(&destination).await?, served_file.bytes.to_vec());
        assert!(std::path::PathBuf::from(format!("{}.crc", destination.display())).is_file());
    }

    Ok(())
}
