use std::{path::PathBuf, sync::Arc};

/// Cross-platform download example
/// This example shows how a single file uses the same group lifecycle as a model.
use download_manager::{
    FileCheck, FileDownloadGroup, FileDownloadGroupPhase, FileDownloadGroupSpec, FileDownloadManager,
    FileDownloadRequest, RelativeFilePath,
};
use kiban::rt::RuntimeHandle;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime_handle = RuntimeHandle::current();

    let manager: Arc<dyn FileDownloadManager> =
        Arc::from(<dyn FileDownloadManager>::system_default(runtime_handle).await?);

    let url = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json".to_string();

    let destination = PathBuf::from("/tmp/test_tokenizer.json");
    if destination.exists() {
        std::fs::remove_file(&destination)?;
    }

    println!("Opening one-file download group...");
    let destination_root = destination.parent().expect("example destination has a parent");
    let relative_path = RelativeFilePath::try_from(
        destination.file_name().expect("example destination has a file name").to_string_lossy().as_ref(),
    )?;
    let spec = FileDownloadGroupSpec::new(
        destination_root,
        [FileDownloadRequest::new(url, relative_path, FileCheck::None, None)],
    )?;
    let group = FileDownloadGroup::open(manager, spec).await?;

    println!("Starting download...");
    let mut progress_stream = group.subscribe();
    let attempt = group.download().await?;

    use tokio_stream::StreamExt;
    while let Some(state) = progress_stream.next().await {
        println!("Progress: {} / {:?} bytes ({:?})", state.downloaded_bytes, state.total_bytes, state.phase);

        if matches!(state.phase, FileDownloadGroupPhase::Downloaded | FileDownloadGroupPhase::Error) {
            break;
        }
    }

    attempt.wait().await?;
    println!("Download finished!");
    Ok(())
}
