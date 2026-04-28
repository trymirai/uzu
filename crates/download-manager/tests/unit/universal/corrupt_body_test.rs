use download_manager::{FileCheck, FileDownloadManagerType, create_download_manager};
use tokio::runtime::Handle as TokioHandle;

use crate::common::{Behavior, MockRegistry, PhaseKind, error_message, wait_for_phase_kind};

#[tokio::test(flavor = "multi_thread")]
async fn test_universal_corrupt_body_fails_crc() {
    let registry = MockRegistry::start_with(Behavior::CorruptBody).await;
    let tokenizer = registry.file("tokenizer.json");
    let temp_dir = tempfile::tempdir().unwrap();
    let destination = temp_dir.path().join(&tokenizer.file.name);
    let manager = create_download_manager(FileDownloadManagerType::Universal, None, TokioHandle::current())
        .await
        .unwrap();
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
    let state = wait_for_phase_kind(&task, &mut progress, PhaseKind::Error).await;
    let message = error_message(state);
    assert!(
        message.contains("CRC") || message.contains("checksum"),
        "unexpected error: {message}"
    );
}
