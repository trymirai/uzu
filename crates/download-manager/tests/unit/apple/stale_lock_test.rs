use chrono::{Duration as ChronoDuration, Utc};
use download_manager::{LockFileInfo, LockFileState, acquire_lock, check_lock_file, release_lock};

use crate::common::{
    mock_download_server::{MockFile, RouteBehavior},
    scenarios::{DownloadScenario, ManagerKind},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_stale_lock_can_be_reacquired() {
    let scenario = DownloadScenario::new(
        ManagerKind::Apple,
        MockFile::tokenizer("download-manager/apple-stale-lock"),
        RouteBehavior::Normal,
    )
    .await;
    let lock_path = scenario.lock_path();
    let stale_lock = LockFileInfo {
        manager_id: "other-manager".to_string(),
        acquired_at: Utc::now() - ChronoDuration::hours(2),
        process_id: 999_999,
    };
    tokio::fs::write(&lock_path, serde_json::to_vec(&stale_lock).expect("failed to serialize lock"))
        .await
        .expect("failed to seed stale lock");

    let lock_state = check_lock_file(&lock_path, scenario.manager.manager_id(), std::process::id());
    assert!(matches!(lock_state, LockFileState::Stale(_)));

    let handle = tokio::runtime::Handle::current();
    acquire_lock(&handle, &lock_path, scenario.manager.manager_id()).await.expect("failed to reacquire stale lock");
    assert!(matches!(
        check_lock_file(&lock_path, scenario.manager.manager_id(), std::process::id()),
        LockFileState::OwnedByUs(_)
    ));
    release_lock(&handle, &lock_path).await.expect("failed to release lock");
}
