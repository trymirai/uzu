use std::sync::Arc;

use download_manager::FileDownloadManagerType;
use mock_registry::MockRegistry;
use rstest::rstest;
use tokio::runtime::Handle as TokioHandle;

use crate::common::{test_storage::TestStorage, tracing_setup::init_test_tracing};

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_storage_cancel_preserves_unrelated_files_in_cache_path(
    #[case] download_manager_type: FileDownloadManagerType,
) -> Result<(), Box<dyn std::error::Error>> {
    init_test_tracing();
    let registry = MockRegistry::start().await?;
    let model = registry.models.first().ok_or_else(|| std::io::Error::other("mock registry must include a model"))?;
    let test_storage =
        TestStorage::with_models_and_manager(TokioHandle::current(), vec![model.clone()], download_manager_type)
            .await?;
    let item = Arc::new(
        test_storage.storage.get(&model.identifier).await.ok_or_else(|| format!("missing item: {}", model.identifier))?,
    );

    tokio::fs::create_dir_all(&item.cache_path).await?;
    let unrelated = item.cache_path.join("user-notes.txt");
    tokio::fs::write(&unrelated, b"data not part of the model").await?;
    assert!(unrelated.exists(), "precondition: unrelated file is on disk before cancel");

    item.cancel().await?;

    assert!(
        unrelated.exists(),
        "Item::cancel must not delete files it does not own; remove_dir_all(&self.cache_path) blasts the entire directory"
    );

    Ok(())
}
