use std::sync::Arc;

use download_manager::{
    DownloadError, FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase, HttpDownloadRequest,
    compute_download_id, traits::DownloadConfig,
};
use kiban::rt::RuntimeHandle;
use rstest::rstest;

use crate::common::{Behavior, MockRegistry, wait_for_phase};

#[tokio::test(flavor = "multi_thread")]
async fn fresh_start_failure_is_published_in_the_atomic_snapshot() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let destination_parent = temp_dir.path().join("models");
    tokio::fs::create_dir(&destination_parent).await?;
    let destination = destination_parent.join("model.bin");
    let artifact_root = temp_dir.path().join("artifacts");
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    let task = manager
        .http_file_download_task_with_artifact_root(
            HttpDownloadRequest::get("https://example.invalid/model.bin"),
            &destination,
            FileCheck::None,
            None,
            &artifact_root,
        )
        .await?;

    tokio::fs::remove_dir(&destination_parent).await?;
    tokio::fs::write(&destination_parent, b"file").await?;

    let error = task.download().await.expect_err("a destination directory cannot replace a regular file");
    let snapshot_receiver = task.snapshot_receiver();
    let snapshot = snapshot_receiver.borrow().clone();

    assert!(matches!(snapshot.state.phase, FileDownloadPhase::Error(_)));
    assert_eq!(snapshot.failure, Some(error));
    tokio::fs::remove_file(destination_parent).await?;
    Ok(())
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn destructive_cancel_rejects_a_late_artifact_symlink() -> Result<(), Box<dyn std::error::Error>> {
    use std::os::unix::fs::symlink;

    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let directory = tempfile::tempdir()?;
    let outside = tempfile::tempdir()?;
    let destination = directory.path().join("models/tokenizer.json");
    let artifact_root = directory.path().join("manager/member");
    let moved_artifact_root = directory.path().join("manager/member-before-symlink");
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    let task = manager
        .http_file_download_task_with_artifact_root(
            HttpDownloadRequest::get(&tokenizer.file.url),
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
            &artifact_root,
        )
        .await?;

    task.download().await?;
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !artifact_root.join("download.part").exists() {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    tokio::fs::rename(&artifact_root, &moved_artifact_root).await?;
    symlink(outside.path(), &artifact_root)?;
    for name in ["download.part", "download.resume_data", "installing", "integrity.json", "recovery.json"] {
        tokio::fs::write(outside.path().join(name), b"keep").await?;
    }

    let error = task.cancel_and_delete().await.expect_err("late artifact symlink must block destructive cleanup");

    assert!(matches!(error, DownloadError::CleanupFailures { .. }));
    for name in ["download.part", "download.resume_data", "installing", "integrity.json", "recovery.json"] {
        assert_eq!(tokio::fs::read(outside.path().join(name)).await?, b"keep");
    }
    tokio::fs::remove_file(artifact_root).await?;
    Ok(())
}

#[cfg(unix)]
#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn pause_quiesces_instead_of_writing_through_a_late_artifact_symlink(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    use std::os::unix::fs::symlink;

    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let directory = tempfile::tempdir()?;
    let outside = tempfile::tempdir()?;
    let destination = directory.path().join("models/tokenizer.json");
    let artifact_root = directory.path().join("manager/member");
    let moved_artifact_root = directory.path().join("manager/member-before-symlink");
    let manager = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await?;
    let task = manager
        .http_file_download_task_with_artifact_root(
            HttpDownloadRequest::get(&tokenizer.file.url),
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
            &artifact_root,
        )
        .await?;

    task.download().await?;
    assert!(matches!(task.state().await.phase, FileDownloadPhase::Downloading));
    tokio::fs::rename(&artifact_root, &moved_artifact_root).await?;
    symlink(outside.path(), &artifact_root)?;
    let outside_resume = outside.path().join("download.resume_data");
    tokio::fs::write(&outside_resume, b"keep").await?;

    let error = task.pause().await.expect_err("late artifact symlink must prevent resume-data writes");

    assert!(matches!(error, DownloadError::CleanupFailures { .. }));
    assert_eq!(tokio::fs::read(&outside_resume).await?, b"keep");
    tokio::fs::remove_file(artifact_root).await?;
    Ok(())
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn inactive_replacement_rejects_a_late_artifact_symlink() -> Result<(), Box<dyn std::error::Error>> {
    use std::os::unix::fs::symlink;

    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let directory = tempfile::tempdir()?;
    let outside = tempfile::tempdir()?;
    let destination = directory.path().join("models/tokenizer.json");
    let artifact_root = directory.path().join("manager/member");
    let moved_artifact_root = directory.path().join("manager/member-before-symlink");
    let manager = <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, RuntimeHandle::current()).await?;
    let task = manager
        .http_file_download_task_with_artifact_root(
            HttpDownloadRequest::get(&tokenizer.file.url),
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
            &artifact_root,
        )
        .await?;

    task.download().await?;
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !artifact_root.join("download.part").exists() {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    task.pause().await?;
    tokio::fs::rename(&artifact_root, &moved_artifact_root).await?;
    symlink(outside.path(), &artifact_root)?;
    for name in ["download.part", "recovery.json", "recovery.tmp"] {
        tokio::fs::write(outside.path().join(name), b"keep").await?;
    }

    let replacement_url = format!("{}?replacement=1", tokenizer.file.url);
    let error = manager
        .http_file_download_task_with_artifact_root(
            HttpDownloadRequest::get(replacement_url),
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
            &artifact_root,
        )
        .await
        .expect_err("late artifact symlinks must block inactive replacement cleanup");

    assert!(matches!(error, DownloadError::CleanupFailures { .. }));
    for name in ["download.part", "recovery.json", "recovery.tmp"] {
        assert_eq!(tokio::fs::read(outside.path().join(name)).await?, b"keep");
    }
    tokio::fs::remove_file(artifact_root).await?;
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_task_creation_returns_same_task(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempfile::tempdir().unwrap();
    let destination = temp_dir.path().join(&tokenizer.file.name);
    let manager = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await.unwrap();

    let (first, second) = tokio::join!(
        manager.file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        ),
        manager.file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        ),
    );
    let first = first.unwrap();
    let second = second.unwrap();

    assert!(Arc::ptr_eq(&first, &second));
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn inactive_task_is_replaced_when_url_changes(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempfile::tempdir().unwrap();
    let destination = temp_dir.path().join(&tokenizer.file.name);
    let manager = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await.unwrap();

    let first = manager
        .file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await
        .unwrap();
    let second = manager
        .file_download_task(
            "http://example.invalid/different-url",
            &destination,
            FileCheck::None,
            Some(tokenizer.file.size as u64),
        )
        .await?;

    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(second.source_url(), "http://example.invalid/different-url");
    assert!(matches!(first.download().await, Err(DownloadError::TaskStopped)));
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn inactive_task_is_replaced_when_expected_bytes_change(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempfile::tempdir().unwrap();
    let destination = temp_dir.path().join(&tokenizer.file.name);
    let manager = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await.unwrap();

    let first = manager
        .file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await
        .unwrap();
    let second = manager
        .file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64 + 1),
        )
        .await?;

    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(second.expected_bytes(), Some(tokenizer.file.size as u64 + 1));
    assert!(matches!(first.download().await, Err(DownloadError::TaskStopped)));
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn active_task_with_different_config_is_not_replaced(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempfile::tempdir()?;
    let destination = temp_dir.path().join(&tokenizer.file.name);
    let manager = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await?;
    let task = manager
        .file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await?;
    let mut progress = task.progress().await?;
    task.download().await?;
    wait_for_phase(&task, &mut progress, |phase| matches!(phase, FileDownloadPhase::Downloading)).await;

    let conflicting_task = manager
        .file_download_task(
            "http://example.invalid/different-url",
            &destination,
            FileCheck::None,
            Some(tokenizer.file.size as u64),
        )
        .await;

    assert!(matches!(conflicting_task, Err(DownloadError::ConflictingConfig(_))));
    assert!(matches!(task.state().await.phase, FileDownloadPhase::Downloading));
    manager.remove_file_task(task.download_id()).await?;
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal, "part")]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple, "resume_data"))]
#[tokio::test(flavor = "multi_thread")]
async fn replacing_paused_task_removes_only_its_resume_artifact(
    #[case] download_manager_type: FileDownloadManagerType,
    #[case] resume_artifact_extension: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempfile::tempdir()?;
    let destination = temp_dir.path().join(&tokenizer.file.name);
    let resume_artifact = DownloadConfig::resume_artifact_path_for(
        &destination,
        compute_download_id(&destination),
        resume_artifact_extension,
    );
    let manager = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await?;
    let task = manager
        .file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await?;
    let mut progress = task.progress().await?;
    task.download().await?;
    wait_for_phase(&task, &mut progress, |phase| matches!(phase, FileDownloadPhase::Downloading)).await;
    if resume_artifact_extension == "part" {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while !resume_artifact.exists() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await?;
    }
    task.pause().await?;
    assert!(resume_artifact.exists());

    let replacement = manager
        .file_download_task(
            "http://example.invalid/different-url",
            &destination,
            FileCheck::None,
            Some(tokenizer.file.size as u64),
        )
        .await?;

    assert!(!Arc::ptr_eq(&task, &replacement));
    assert!(!resume_artifact.exists());
    assert!(matches!(replacement.state().await.phase, FileDownloadPhase::NotDownloaded));
    assert!(matches!(task.download().await, Err(DownloadError::TaskStopped)));
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn replacing_downloaded_task_preserves_destination(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempfile::tempdir()?;
    let destination = temp_dir.path().join(&tokenizer.file.name);
    let manager = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await?;
    let task = manager
        .file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await?;
    task.download().await?;
    task.wait().await;
    assert!(matches!(task.state().await.phase, FileDownloadPhase::Downloaded));
    let contents_before_replacement = tokio::fs::read(&destination).await?;

    let replacement = manager
        .file_download_task(
            "http://example.invalid/different-url",
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await?;

    assert!(!Arc::ptr_eq(&task, &replacement));
    assert_eq!(tokio::fs::read(&destination).await?, contents_before_replacement);
    assert!(matches!(replacement.state().await.phase, FileDownloadPhase::Downloaded));
    assert!(matches!(task.download().await, Err(DownloadError::TaskStopped)));
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn download_and_config_replacement_are_serialized_by_actor(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempfile::tempdir()?;
    let destination = temp_dir.path().join(&tokenizer.file.name);
    let manager = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await?;
    let task = manager
        .file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await?;

    let (download_result, replacement_result) = tokio::join!(
        task.download(),
        manager.file_download_task(
            "http://example.invalid/different-url",
            &destination,
            FileCheck::None,
            Some(tokenizer.file.size as u64),
        ),
    );

    match (download_result, replacement_result) {
        (Ok(()), Err(DownloadError::ConflictingConfig(_))) => {
            assert!(matches!(task.state().await.phase, FileDownloadPhase::Downloading));
        },
        (Err(DownloadError::TaskStopped | DownloadError::ChannelClosed), Ok(replacement)) => {
            assert!(!Arc::ptr_eq(&task, &replacement));
            assert!(matches!(replacement.state().await.phase, FileDownloadPhase::NotDownloaded));
        },
        (download_result, replacement_result) => {
            panic!("unexpected race outcome: download={download_result:?}, replacement={replacement_result:?}")
        },
    }

    manager.remove_file_task(task.download_id()).await?;
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn separate_managers_in_same_process_cannot_share_destination_lock(
    #[case] download_manager_type: FileDownloadManagerType
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempfile::tempdir().unwrap();
    let destination = temp_dir.path().join(&tokenizer.file.name);

    let manager_a = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await.unwrap();
    let task_a = manager_a
        .file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await
        .unwrap();
    let mut progress_a = task_a.progress().await.unwrap();
    task_a.download().await.unwrap();
    wait_for_phase(&task_a, &mut progress_a, |phase| matches!(phase, FileDownloadPhase::Downloading)).await;

    let manager_b = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await.unwrap();
    let task_b = manager_b
        .file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await
        .unwrap();

    assert!(matches!(task_b.state().await.phase, FileDownloadPhase::LockedByOther(_)));

    drop(task_b);
    drop(manager_b);

    let progress_a_after_b_dropped = task_a.state().await.phase;
    assert!(!matches!(progress_a_after_b_dropped, FileDownloadPhase::Error(_)));
    Ok(())
}
