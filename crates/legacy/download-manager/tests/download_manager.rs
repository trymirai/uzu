mod common;

use std::{path::Path, sync::Arc, time::Duration};

use download_manager::{
    DestinationLock, DownloadError, DownloadManager, DownloadManagerType, DownloadPhase, DownloadState, DownloadTask,
    DownloadTaskRequest, LockError,
};
use kiban::rt::RuntimeHandle;
use rstest::rstest;
use tokio::time::timeout;
use uuid::Uuid;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

use crate::common::{Behavior, MockRegistry, artifact_path, file_request, foreign_lock, model_request, wait_for_state};

fn manager(kind: DownloadManagerType) -> DownloadManager {
    DownloadManager::new(kind, RuntimeHandle::current())
}

fn assert_sequential(
    task: &DownloadTask,
    state: &DownloadState,
    last_downloaded_bytes: &mut i64,
) {
    let downloading =
        task.subtasks().iter().filter(|child| matches!(child.state().phase, DownloadPhase::Downloading {})).count();
    assert!(downloading <= 1, "{downloading} files downloading at once");
    assert!(state.downloaded_bytes >= *last_downloaded_bytes, "downloaded bytes went backwards");
    *last_downloaded_bytes = state.downloaded_bytes;
}

#[rstest]
#[case::universal(DownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::native(DownloadManagerType::Native))]
#[tokio::test(flavor = "multi_thread")]
async fn model_lifecycle(#[case] kind: DownloadManagerType) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let directory = tempfile::tempdir()?;
    let manager = manager(kind);
    let request = model_request(&registry, directory.path())?;
    let total_bytes: i64 = registry.files.iter().map(|served| served.file.size).sum();
    let mut last_downloaded_bytes = 0;

    let model = manager.download_task(request.clone()).await?;
    assert!(Arc::ptr_eq(&model, &manager.download_task(request.clone()).await?));
    let bundle_request = DownloadTaskRequest::group()
        .destination(directory.path().join("bundle"))
        .subrequests(vec![request.clone()])
        .build();
    let bundle = manager.download_task(bundle_request).await?;
    assert!(Arc::ptr_eq(&bundle.subtasks()[0], &model));
    assert_eq!(model.state().total_bytes, total_bytes);
    assert_eq!(bundle.state().total_bytes, total_bytes);
    assert_eq!(model.state().phase, DownloadPhase::NotDownloaded {});

    let mut progress = model.progress();
    model.download().await?;
    wait_for_state(&model, &mut progress, |state| {
        assert_sequential(&model, state, &mut last_downloaded_bytes);
        matches!(state.phase, DownloadPhase::Downloading {}) && state.downloaded_bytes >= total_bytes / 4
    })
    .await;

    model.pause().await?;
    let paused = wait_for_state(&model, &mut progress, |state| matches!(state.phase, DownloadPhase::Paused {})).await;
    assert!(paused.downloaded_bytes >= total_bytes / 4);
    assert_eq!(bundle.state().phase, DownloadPhase::Paused {});

    drop(progress);
    drop(bundle);
    drop(model);
    let model = manager.download_task(request).await?;
    assert_eq!(model.state().phase, DownloadPhase::Paused {});
    assert_eq!(model.state().downloaded_bytes, paused.downloaded_bytes);

    let mut progress = model.progress();
    model.download().await?;
    let downloaded = wait_for_state(&model, &mut progress, |state| {
        assert_sequential(&model, state, &mut last_downloaded_bytes);
        matches!(state.phase, DownloadPhase::Downloaded {})
    })
    .await;
    assert_eq!(downloaded.downloaded_bytes, total_bytes);
    assert_eq!(downloaded.total_bytes, total_bytes);
    for served in registry.files.iter() {
        let destination = directory.path().join(&served.file.name);
        assert_eq!(tokio::fs::read(&destination).await?, served.bytes.to_vec());
        assert!(artifact_path(&destination, "crc").is_file());
    }

    model.delete().await?;
    assert_eq!(model.state().phase, DownloadPhase::NotDownloaded {});
    assert_eq!(model.state().downloaded_bytes, 0);
    for served in registry.files.iter() {
        let destination = directory.path().join(&served.file.name);
        assert!(!destination.exists());
        assert!(!artifact_path(&destination, "crc").exists());
        assert!(!artifact_path(&destination, "lock").exists());
    }

    let first = registry.files.first().expect("mock registry serves files");
    let destination = directory.path().join(&first.file.name);
    let expected_bytes = Some(first.file.size as u64);
    let conflicting_url =
        manager.download_task(file_request("http://example.invalid/other", &destination, None, expected_bytes)).await;
    assert!(matches!(conflicting_url, Err(DownloadError::ConflictingConfig(_))));
    let conflicting_size =
        manager.download_task(file_request(&first.file.url, &destination, Some(first.crc32c()?), Some(1))).await;
    assert!(matches!(conflicting_size, Err(DownloadError::ConflictingConfig(_))));
    let same = file_request(&first.file.url, &destination, Some(first.crc32c()?), expected_bytes);
    let (task_a, task_b) = tokio::join!(manager.download_task(same.clone()), manager.download_task(same));
    assert!(Arc::ptr_eq(&task_a?, &task_b?));
    Ok(())
}

#[rstest]
#[case::universal(DownloadManagerType::Universal, "part")]
#[cfg_attr(target_vendor = "apple", case::native(DownloadManagerType::Native, "resume_data"))]
#[tokio::test(flavor = "multi_thread")]
async fn startup_reconciliation(
    #[case] kind: DownloadManagerType,
    #[case] resume_artifact_extension: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start().await?;
    let served = registry.file("config.json")?;
    let crc = served.crc32c()?;
    let size = Some(served.file.size as u64);
    let manager = manager(kind);
    let request = |destination: &Path| file_request(&served.file.url, destination, Some(crc.clone()), size);

    let valid = tempfile::tempdir()?;
    let destination = valid.path().join(&served.file.name);
    let artifact = artifact_path(&destination, resume_artifact_extension);
    tokio::fs::write(&destination, served.bytes.as_ref()).await?;
    tokio::fs::write(&artifact, b"partial").await?;
    let task = manager.download_task(request(&destination)).await?;
    assert_eq!(task.state().phase, DownloadPhase::Downloaded {});
    assert!(!artifact.exists());
    let receipt: serde_json::Value =
        serde_json::from_str(&tokio::fs::read_to_string(artifact_path(&destination, "crc")).await?)?;
    assert_eq!(receipt["version"].as_u64(), Some(1));
    assert_eq!(receipt["crc"].as_str(), Some(crc.as_str()));
    assert_eq!(receipt["file_size"].as_u64(), size);

    let mut changed_bytes = served.bytes.to_vec();
    changed_bytes[0] = changed_bytes[0].wrapping_add(1);
    let stale_receipt = serde_json::to_vec(&serde_json::json!({
        "version": 1,
        "crc": crc.clone(),
        "file_size": served.file.size,
        "modified_unix_seconds": 0,
        "modified_nanos": 0,
    }))?;
    for cache in [crc.clone().into_bytes(), stale_receipt] {
        let stale = tempfile::tempdir()?;
        let destination = stale.path().join(&served.file.name);
        tokio::fs::write(&destination, &changed_bytes).await?;
        tokio::fs::write(artifact_path(&destination, "crc"), cache).await?;
        let task = manager.download_task(request(&destination)).await?;
        assert_eq!(task.state().phase, DownloadPhase::NotDownloaded {});
        assert!(!destination.exists());
    }

    let folder = tempfile::tempdir()?;
    let destination = folder.path().join(&served.file.name);
    tokio::fs::create_dir(&destination).await?;
    let task = manager.download_task(file_request(&served.file.url, &destination, None, None)).await?;
    assert_eq!(task.state().phase, DownloadPhase::NotDownloaded {});

    let locked = tempfile::tempdir()?;
    let destination = locked.path().join(&served.file.name);
    let artifact = artifact_path(&destination, resume_artifact_extension);
    tokio::fs::write(&destination, b"corrupt").await?;
    tokio::fs::write(artifact_path(&destination, "crc"), &crc).await?;
    tokio::fs::write(&artifact, b"partial").await?;
    let lock = foreign_lock(&destination).await;
    let group = manager
        .download_task(
            DownloadTaskRequest::group()
                .destination(locked.path())
                .subrequests(vec![file_request(
                    &served.file.url,
                    Path::new(&served.file.name),
                    Some(crc.clone()),
                    size,
                )])
                .build(),
        )
        .await?;
    assert!(matches!(group.state().phase, DownloadPhase::Locked { .. }));
    assert!(matches!(group.delete().await, Err(DownloadError::Lock(LockError::LockedByOther { .. }))));
    assert!(destination.exists());
    assert!(artifact_path(&destination, "crc").exists());
    assert!(artifact.exists());
    let mut progress = group.progress();
    if kind == DownloadManagerType::Universal {
        tokio::fs::write(&artifact, b"partial-and-more").await?;
        wait_for_state(&group, &mut progress, |state| state.downloaded_bytes == b"partial-and-more".len() as i64).await;
    }
    drop(lock);
    wait_for_state(&group, &mut progress, |state| matches!(state.phase, DownloadPhase::Paused {})).await;
    assert!(!destination.exists());
    assert!(artifact.exists());

    let paused = tempfile::tempdir()?;
    let destination = paused.path().join(&served.file.name);
    let artifact = artifact_path(&destination, resume_artifact_extension);
    tokio::fs::write(&artifact, b"partial").await?;
    let task = manager.download_task(request(&destination)).await?;
    assert_eq!(task.state().phase, DownloadPhase::Paused {});
    task.delete().await?;
    assert_eq!(task.state().phase, DownloadPhase::NotDownloaded {});
    assert!(!artifact.exists());

    let empty = tempfile::tempdir()?;
    let destination = empty.path().join(&served.file.name);
    tokio::fs::write(artifact_path(&destination, resume_artifact_extension), b"").await?;
    let task = manager.download_task(request(&destination)).await?;
    assert_eq!(task.state().phase, DownloadPhase::Paused {});
    let mut progress = task.progress();
    task.download().await?;
    wait_for_state(&task, &mut progress, |state| matches!(state.phase, DownloadPhase::Downloaded {})).await;
    assert_eq!(tokio::fs::read(&destination).await?, served.bytes.to_vec());
    Ok(())
}

#[rstest]
#[case::universal(DownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::native(DownloadManagerType::Native))]
#[tokio::test(flavor = "multi_thread")]
async fn failures(#[case] kind: DownloadManagerType) -> Result<(), Box<dyn std::error::Error>> {
    let manager = manager(kind);

    let corrupt = MockRegistry::start_with(Behavior::CORRUPT_BODY).await?;
    let served = corrupt.file("tokenizer.json")?;
    let directory = tempfile::tempdir()?;
    let destination = directory.path().join(&served.file.name);
    let task = manager
        .download_task(file_request(
            &served.file.url,
            &destination,
            Some(served.crc32c()?),
            Some(served.file.size as u64),
        ))
        .await?;
    let mut progress = task.progress();
    task.download().await?;
    let state = wait_for_state(&task, &mut progress, |state| matches!(state.phase, DownloadPhase::Error { .. })).await;
    let DownloadPhase::Error {
        message,
    } = state.phase
    else {
        panic!("expected an error phase, got {:?}", state.phase)
    };
    assert!(message.contains("CRC"), "unexpected error: {message}");
    task.delete().await?;
    assert_eq!(task.state().phase, DownloadPhase::NotDownloaded {});

    let truncated = MockRegistry::start_with(Behavior::TRUNCATE_BODY).await?;
    let served = truncated.file("config.json")?;
    let destination = directory.path().join(&served.file.name);
    let task = manager
        .download_task(file_request(&served.file.url, &destination, None, Some(served.file.size as u64)))
        .await?;
    let mut progress = task.progress();
    task.download().await?;
    let state = wait_for_state(&task, &mut progress, |state| {
        matches!(state.phase, DownloadPhase::Downloaded {} | DownloadPhase::Error { .. })
    })
    .await;
    assert!(
        matches!(state.phase, DownloadPhase::Error { .. }),
        "truncated body must surface as Error, got {:?}",
        state.phase
    );
    Ok(())
}

#[rstest]
#[case::universal(DownloadManagerType::Universal, "part")]
#[cfg_attr(target_vendor = "apple", case::native(DownloadManagerType::Native, "resume_data"))]
#[tokio::test(flavor = "multi_thread")]
async fn shutdown_and_locks(
    #[case] kind: DownloadManagerType,
    #[case] resume_artifact_extension: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
    let served = registry.file("tokenizer.json")?;
    let directory = tempfile::tempdir()?;
    let destination = directory.path().join(&served.file.name);
    let lock = artifact_path(&destination, "lock");
    let artifact = artifact_path(&destination, resume_artifact_extension);
    let request = || {
        file_request(&served.file.url, &destination, Some(served.crc32c().expect("crc")), Some(served.file.size as u64))
    };

    let manager_a = manager(kind);
    let task_a = manager_a.download_task(request()).await?;
    let mut progress_a = task_a.progress();
    task_a.download().await?;
    wait_for_state(&task_a, &mut progress_a, |state| {
        matches!(state.phase, DownloadPhase::Downloading {}) && state.downloaded_bytes > 0
    })
    .await;
    assert!(lock.exists());

    let manager_b = manager(kind);
    let task_b = manager_b.download_task(request()).await?;
    assert!(matches!(task_b.state().phase, DownloadPhase::Locked { .. }));
    drop(task_b);
    drop(manager_b);
    assert_eq!(task_a.state().phase, DownloadPhase::Downloading {});

    drop(progress_a);
    drop(task_a);
    drop(manager_a);
    timeout(Duration::from_secs(2), async {
        while DestinationLock::foreign_owner(&destination, "probe", Uuid::new_v4()).await.is_some() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await?;
    assert!(artifact.exists());
    let size_at_release = artifact.metadata()?.len();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(artifact.metadata()?.len(), size_at_release);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn universal_resume() -> Result<(), Box<dyn std::error::Error>> {
    let manager = manager(DownloadManagerType::Universal);
    let full_bytes: &[u8] = b"abcdefghij";
    let partial_bytes: &[u8] = b"abcde";

    let ignoring_range = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(full_bytes))
        .mount(&ignoring_range)
        .await;
    let directory = tempfile::tempdir()?;
    let destination = directory.path().join("model.bin");
    tokio::fs::write(artifact_path(&destination, "part"), partial_bytes).await?;
    let task = manager
        .download_task(file_request(
            &format!("{}/model.bin", ignoring_range.uri()),
            &destination,
            None,
            Some(full_bytes.len() as u64),
        ))
        .await?;
    assert_eq!(task.state().phase, DownloadPhase::Paused {});
    let mut progress = task.progress();
    task.download().await?;
    wait_for_state(&task, &mut progress, |state| matches!(state.phase, DownloadPhase::Downloaded {})).await;
    assert_eq!(tokio::fs::read(&destination).await?, full_bytes);

    let misaligned = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/model.bin"))
        .respond_with(
            ResponseTemplate::new(206)
                .set_body_bytes(partial_bytes)
                .insert_header("Content-Range", format!("bytes 0-{}/{}", full_bytes.len() - 1, full_bytes.len())),
        )
        .mount(&misaligned)
        .await;
    let directory = tempfile::tempdir()?;
    let destination = directory.path().join("model.bin");
    tokio::fs::write(artifact_path(&destination, "part"), partial_bytes).await?;
    let task = manager
        .download_task(file_request(
            &format!("{}/model.bin", misaligned.uri()),
            &destination,
            None,
            Some(full_bytes.len() as u64),
        ))
        .await?;
    assert_eq!(task.state().phase, DownloadPhase::Paused {});
    let mut progress = task.progress();
    task.download().await?;
    let state = wait_for_state(&task, &mut progress, |state| matches!(state.phase, DownloadPhase::Error { .. })).await;
    let DownloadPhase::Error {
        message,
    } = state.phase
    else {
        panic!("expected an error phase, got {:?}", state.phase)
    };
    assert!(message.contains("starting at"), "unexpected error: {message}");
    assert!(!destination.exists());
    Ok(())
}
