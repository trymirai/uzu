use std::process::id as process_id;

use chrono::{Duration as ChronoDuration, Utc};
use download_manager::{
    FileDownloadManagerType, LockFileInfo, LockFileState, acquire_lock, check_lock_file, release_lock,
    create_download_manager,
};
use serde_json::to_vec;
use tokio::{fs::write as tokio_write, runtime::Handle as TokioHandle};

use crate::common::{
    mock_download_server::RouteBehavior,
    scenarios::DownloadTestContext,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_apple_stale_lock_can_be_reacquired() {
    let context = DownloadTestContext::new(
        "tokenizer.json",
        RouteBehavior::Normal,
    )
    .await;
    let manager = create_download_manager(
        FileDownloadManagerType::Apple,
        Some("apple-stale-lock".to_string()),
        TokioHandle::current(),
    )
    .await
    .expect("failed to create download manager");
    let lock_path = context.lock_path();
    let stale_lock = LockFileInfo {
        manager_id: "other-manager".to_string(),
        acquired_at: Utc::now() - ChronoDuration::hours(2),
        process_id: 999_999,
    };
    tokio_write(&lock_path, to_vec(&stale_lock).expect("failed to serialize lock"))
        .await
        .expect("failed to seed stale lock");

    let lock_state = check_lock_file(&lock_path, manager.manager_id(), process_id());
    assert!(matches!(lock_state, LockFileState::Stale(_)));

    let handle = TokioHandle::current();
    acquire_lock(&handle, &lock_path, manager.manager_id()).await.expect("failed to reacquire stale lock");
    assert!(matches!(
        check_lock_file(&lock_path, manager.manager_id(), process_id()),
        LockFileState::OwnedByUs(_)
    ));
    release_lock(&handle, &lock_path).await.expect("failed to release lock");
}
