#![cfg(not(target_family = "wasm"))]

use std::{io, path::PathBuf};

use download_manager::FileDownloadManagerType;
use mock_registry::MockRegistry;
use rstest::rstest;
use tokio::time::{Duration, timeout};
use uzu::{
    engine::{Engine, EngineConfig},
    registry::FixedRegistry,
    storage::types::DownloadPhase,
};

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_engine_downloads_mock_registry_model(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let model = registry.models.first().cloned().ok_or_else(|| io::Error::other("mock registry has no models"))?;
    for served_file in registry.files.iter() {
        assert!(
            served_file.file.url.starts_with("http://127.0.0.1:"),
            "mock registry must serve files from localhost, got {}",
            served_file.file.url,
        );
    }

    let temporary_directory = tempfile::tempdir()?;
    let engine_config = EngineConfig::default().with_allow_ollama_usage(false).with_allow_lmstudio_usage(false);
    let engine = Engine::new_without_default_registries(
        engine_config,
        download_manager_type,
        Some(temporary_directory.path().to_path_buf()),
    )
    .await?;
    engine
        .add_registry(Box::new(FixedRegistry::new(
            "mock_registry".to_string(),
            registry.models.iter().cloned().collect(),
        )))
        .await?;

    let resolved_model = engine
        .model(model.identifier.clone())
        .await?
        .ok_or_else(|| io::Error::other(format!("model not found: {}", model.identifier)))?;

    let stream = engine.download(&resolved_model).await?;
    let mut final_update = None;
    timeout(Duration::from_secs(120), async {
        while let Some(update) = stream.next().await {
            final_update = Some(update);
        }
    })
    .await?;

    let final_state = engine
        .download_state(&resolved_model)
        .await
        .ok_or_else(|| io::Error::other(format!("download state missing: {}", resolved_model.identifier)))?;
    assert!(
        matches!(final_state.phase, DownloadPhase::Downloaded {}),
        "mock model download must finish, got {:?}",
        final_state.phase,
    );

    let expected_total_bytes = registry.files.iter().map(|served_file| served_file.file.size).sum::<i64>();
    assert_eq!(final_state.total_bytes, expected_total_bytes);
    assert_eq!(final_state.downloaded_bytes, expected_total_bytes);
    if let Some(update) = final_update {
        assert_eq!(update.bytes_total, expected_total_bytes);
        assert_eq!(update.bytes_downloaded, expected_total_bytes);
    }

    let model_path = PathBuf::from(
        engine
            .model_path(&resolved_model)
            .await
            .ok_or_else(|| io::Error::other(format!("model path missing: {}", resolved_model.identifier)))?,
    );
    for served_file in registry.files.iter() {
        let destination = model_path.join(&served_file.file.name);
        assert_eq!(tokio::fs::read(&destination).await?, served_file.bytes.to_vec());
    }

    Ok(())
}
