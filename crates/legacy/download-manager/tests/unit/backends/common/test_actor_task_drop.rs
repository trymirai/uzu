use std::{error::Error, sync::Arc, time::Duration};

use download_manager::{DownloadError, FileCheck, FileDownloadManager, FileDownloadManagerType, FileDownloadPhase};
use kiban::rt::RuntimeHandle;
use rstest::rstest;
use tempfile::tempdir;
use tokio::{fs::write as tokio_write, time::sleep as tokio_sleep};

use crate::common::{Behavior, MockRegistry, wait_for_phase};

#[rstest]
#[case::universal(FileDownloadManagerType::Universal, "part")]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple, "resume_data"))]
#[tokio::test(flavor = "multi_thread")]
async fn remove_paused_task_deletes_resume_artifact(
    #[case] download_manager_type: FileDownloadManagerType,
    #[case] resume_artifact_extension: &str,
) -> Result<(), Box<dyn Error>> {
    let temporary_directory = tempdir()?;
    let destination = temporary_directory.path().join("model.bin");
    let resume_artifact = destination.with_added_extension(resume_artifact_extension);
    tokio_write(&resume_artifact, b"partial").await?;

    let manager = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await?;
    let task = manager
        .file_download_task("http://example.invalid/model.bin".into(), &destination, FileCheck::None, Some(100))
        .await?;

    assert!(matches!(task.state().await.phase, FileDownloadPhase::Paused));

    manager.remove_file_task(task.download_id()).await?;

    assert!(!resume_artifact.exists());
    assert!(matches!(task.download().await, Err(DownloadError::TaskStopped)));
    Ok(())
}

#[rstest]
#[case::universal(FileDownloadManagerType::Universal, "part")]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple, "resume_data"))]
#[tokio::test(flavor = "multi_thread")]
async fn dropping_active_download_cancels_backend_before_releasing_lock(
    #[case] download_manager_type: FileDownloadManagerType,
    #[case] resume_artifact_extension: &str,
) -> Result<(), Box<dyn Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let tokenizer = registry.file("tokenizer.json")?;
    let temp_dir = tempdir()?;
    let destination = temp_dir.path().join(&tokenizer.file.name);
    let lock_path = destination.with_added_extension("lock");
    let resume_artifact = destination.with_added_extension(resume_artifact_extension);
    let manager = <dyn FileDownloadManager>::new(download_manager_type, RuntimeHandle::current()).await.unwrap();
    let task = manager
        .file_download_task(
            (&tokenizer.file.url).into(),
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await
        .unwrap();
    let mut progress = task.progress().await.unwrap();
    task.download().await.unwrap();
    wait_for_phase(&task, &mut progress, |phase| matches!(phase, FileDownloadPhase::Downloading)).await;
    assert!(lock_path.exists());

    manager.remove_file_task(task.download_id()).await.unwrap();

    assert!(matches!(task.download().await, Err(DownloadError::TaskStopped)));
    let replacement_task = manager
        .file_download_task(
            (&tokenizer.file.url).into(),
            &destination,
            FileCheck::CRC(tokenizer.crc32c()?),
            Some(tokenizer.file.size as u64),
        )
        .await
        .unwrap();
    assert!(!Arc::ptr_eq(&task, &replacement_task));
    drop(replacement_task);

    let mut lock_released = false;
    for _ in 0..100 {
        if !lock_path.exists() {
            lock_released = true;
            break;
        }
        tokio_sleep(Duration::from_millis(20)).await;
    }
    assert!(lock_released);
    assert!(!resume_artifact.exists());

    let destination_size_at_release = destination.metadata().map(|m| m.len()).unwrap_or(0);
    tokio_sleep(Duration::from_millis(200)).await;
    let destination_size_after = destination.metadata().map(|m| m.len()).unwrap_or(0);
    assert_eq!(destination_size_at_release, destination_size_after);
    Ok(())
}
