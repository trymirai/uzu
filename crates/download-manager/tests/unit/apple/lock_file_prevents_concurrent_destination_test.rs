use crate::common::{
    mock_download_server::{MockFile, RouteBehavior},
    scenarios::{DownloadScenario, ManagerKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_lock_file_prevents_concurrent_destination() {
    let scenario = DownloadScenario::new(
        ManagerKind::Apple,
        MockFile::large_tokenizer("download-manager/apple-lock"),
        RouteBehavior::StallAt {
            byte_offset: 64 * 1024,
        },
    )
    .await;

    scenario.start_download().await;
    scenario.wait_for_bytes(64 * 1024).await;
    assert!(scenario.lock_path().exists(), "download should acquire a destination lock");

    let other_manager = download_manager::create_download_manager(
        download_manager::FileDownloadManagerType::Apple,
        Some("apple-lock-other-manager".to_string()),
        tokio::runtime::Handle::current(),
    )
    .await
    .expect("failed to create second manager");
    let other_task = other_manager
        .file_download_task(
            &scenario.server.url_for_file(&scenario.file),
            &scenario.destination,
            download_manager::FileCheck::CRC(scenario.file.crc32c.clone()),
            Some(scenario.file.size),
        )
        .await
        .expect("failed to create second task");

    let other_state = other_task.state().await;
    assert!(matches!(other_state.phase, download_manager::FileDownloadPhase::LockedByOther(_)));
    scenario.cancel_download().await;
}
