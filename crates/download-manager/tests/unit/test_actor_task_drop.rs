use std::{path::PathBuf, sync::Arc};

use download_manager::{
    DownloadError, FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase, acquire_lock,
    backends::universal::{UniversalBackend, UniversalBackendContext},
    file_download_task_actor::{GenericFileDownloadTask, ProgressCounters, PublicProjection},
    reducer::InitialLifecycleState,
    traits::DownloadConfig,
};
use rstest::rstest;
use tokio::runtime::Handle as TokioHandle;
use uuid::Uuid;

use crate::common::{Behavior, MockRegistry, wait_for_phase};

#[tokio::test(flavor = "multi_thread")]
async fn dropping_task_releases_destination_lock_when_owned() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let destination = temp_dir.path().join("model.bin");
    let lock_path = PathBuf::from(format!("{}.lock", destination.display()));
    let resume_artifact = destination.with_extension("part");
    let manager_id = "test-manager";
    let instance_id = Uuid::new_v4();

    acquire_lock(&lock_path, manager_id, instance_id).await?;
    tokio::fs::write(&resume_artifact, b"partial").await?;
    assert!(lock_path.exists(), "precondition: lock should be present");
    assert!(resume_artifact.exists(), "precondition: resume artifact should be present");

    let task = GenericFileDownloadTask::<UniversalBackend>::spawn(
        Arc::new(DownloadConfig {
            download_id: Uuid::nil(),
            source_url: "http://example.invalid/file".to_string(),
            destination: destination.clone(),
            file_check: FileCheck::None,
            expected_bytes: Some(120),
            manager_id: manager_id.to_string(),
            manager_instance_id: instance_id,
        }),
        Arc::new(UniversalBackendContext::new(TokioHandle::current())),
        InitialLifecycleState::Paused {
            part_path: resume_artifact.clone(),
        },
        PublicProjection::None,
        ProgressCounters::default(),
    )?;

    drop(task);

    let mut lock_released = false;
    for _ in 0..50 {
        if !lock_path.exists() {
            lock_released = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(
        lock_released,
        "lock at {} was not released within 1s after dropping the task",
        lock_path.display(),
    );
    assert!(
        resume_artifact.exists(),
        "passive task drop must preserve resume artifact at {}",
        resume_artifact.display(),
    );
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal, "part")]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple, "resume_data"))]
#[tokio::test(flavor = "multi_thread")]
async fn remove_paused_task_deletes_resume_artifact(
    #[case] download_manager_type: FileDownloadManagerType,
    #[case] resume_artifact_extension: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let temporary_directory = tempfile::tempdir()?;
    let destination = temporary_directory.path().join("model.bin");
    let resume_artifact = destination.with_extension(resume_artifact_extension);
    tokio::fs::write(&resume_artifact, b"partial").await?;

    let manager = <dyn FileDownloadManager>::new(download_manager_type, TokioHandle::current()).await?;
    let task = manager
        .file_download_task("http://example.invalid/model.bin", &destination, FileCheck::None, Some(100))
        .await?;

    assert!(matches!(task.state().await.phase, FileDownloadPhase::Paused));

    manager.remove_file_task(task.download_id()).await?;

    assert!(
        !resume_artifact.exists(),
        "manager removal must delete resume artifact at {}",
        resume_artifact.display(),
    );
    assert!(
        matches!(task.download().await, Err(DownloadError::TaskStopped)),
        "old task handles must be permanently stopped after manager removal",
    );
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal, "part")]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple, "resume_data"))]
#[tokio::test(flavor = "multi_thread")]
async fn dropping_active_download_cancels_backend_before_releasing_lock(
    #[case] download_manager_type: FileDownloadManagerType,
    #[case] resume_artifact_extension: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempfile::tempdir()?;
    let destination = temp_dir.path().join(&tokenizer.file.name);
    let lock_path = PathBuf::from(format!("{}.lock", destination.display()));
    let resume_artifact = destination.with_extension(resume_artifact_extension);
    let manager = <dyn FileDownloadManager>::new(download_manager_type, TokioHandle::current()).await.unwrap();
    let task = manager
        .file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await
        .unwrap();
    let mut progress = task.progress().await.unwrap();
    task.download().await.unwrap();
    wait_for_phase(&task, &mut progress, |phase| matches!(phase, FileDownloadPhase::Downloading)).await;
    assert!(lock_path.exists(), "precondition: lock acquired during Downloading");

    manager.remove_file_task(task.download_id()).await.unwrap();

    assert!(
        matches!(task.download().await, Err(DownloadError::TaskStopped)),
        "old task handles must be permanently stopped after manager removal",
    );
    let replacement_task = manager
        .file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await
        .unwrap();
    assert!(
        !Arc::ptr_eq(&task, &replacement_task),
        "manager removal must evict the stopped task before allowing a replacement to be built",
    );
    drop(replacement_task);

    let mut lock_released = false;
    for _ in 0..100 {
        if !lock_path.exists() {
            lock_released = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(
        lock_released,
        "lock at {} was not released within 2s after dropping an active download",
        lock_path.display(),
    );
    assert!(
        !resume_artifact.exists(),
        "Cancel must remove the resume artifact at {} as part of shutdown — its presence proves the actor never \
         dispatched Cancel and likely just unlinked the lock while the backend was still writing",
        resume_artifact.display(),
    );

    let destination_size_at_release = destination.metadata().map(|m| m.len()).unwrap_or(0);
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let destination_size_after = destination.metadata().map(|m| m.len()).unwrap_or(0);
    assert_eq!(
        destination_size_at_release, destination_size_after,
        "destination changed after the lock was released — backend was still writing",
    );
    Ok(())
}
