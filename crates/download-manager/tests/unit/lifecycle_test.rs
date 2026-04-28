use download_manager::{FileCheck, FileDownloadManagerType, FileDownloadPhase, create_download_manager};
use rstest::rstest;
use tokio::runtime::Handle as TokioHandle;

use crate::common::{MockRegistry, wait_for_phase};

#[rstest]
#[case::universal(FileDownloadManagerType::Universal)]
#[cfg_attr(target_vendor = "apple", case::apple(FileDownloadManagerType::Apple))]
#[tokio::test(flavor = "multi_thread")]
async fn test_download_fresh_completes(#[case] download_manager_type: FileDownloadManagerType) {
    let registry = MockRegistry::start().await;
    let tokenizer = registry.file("tokenizer.json");
    let temp_dir = tempfile::tempdir().unwrap();
    let destination = temp_dir.path().join(&tokenizer.file.name);

    let manager = create_download_manager(download_manager_type, None, TokioHandle::current()).await.unwrap();
    let task = manager
        .file_download_task(
            &tokenizer.file.url,
            &destination,
            FileCheck::CRC(tokenizer.crc32c()),
            Some(tokenizer.file.size as u64),
        )
        .await
        .unwrap();
    let mut progress = task.progress().await.unwrap();

    task.download().await.unwrap();
    let state = wait_for_phase(&task, &mut progress, |phase| matches!(phase, FileDownloadPhase::Downloaded)).await;

    assert_eq!(state.downloaded_bytes, tokenizer.file.size as u64);
    assert_eq!(state.total_bytes, tokenizer.file.size as u64);
    assert_eq!(tokio::fs::read(&destination).await.unwrap(), tokenizer.bytes.to_vec());
    assert!(std::path::PathBuf::from(format!("{}.crc", destination.display())).is_file());
}
