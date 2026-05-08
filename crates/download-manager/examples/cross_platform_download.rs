use std::path::PathBuf;

/// Cross-platform download example
/// This example shows how to use FileDownloadManager trait
/// which works on both Apple and non-Apple platforms
use download_manager::{FileCheck, FileDownloadManager};
use tokio::runtime::Handle;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokio_handle = Handle::current();

    let manager = <dyn FileDownloadManager>::system_default(tokio_handle).await?;

    let url = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json".to_string();

    let destination = PathBuf::from("/tmp/test_tokenizer.json");
    if destination.exists() {
        std::fs::remove_file(&destination)?;
    }

    println!("Creating download task...");
    let task = manager.file_download_task(&url, &destination, FileCheck::None, None).await?;

    println!("Starting download...");
    let mut progress_stream = task.progress().await?;
    task.download().await?;

    use tokio_stream::StreamExt;
    while let Some(Ok(state)) = progress_stream.next().await {
        println!("Progress: {} / {} bytes ({:?})", state.downloaded_bytes, state.total_bytes, state.phase);

        if matches!(
            state.phase,
            download_manager::FileDownloadPhase::Downloaded | download_manager::FileDownloadPhase::Error(_)
        ) {
            break;
        }
    }

    task.wait().await;
    println!("Download finished!");
    Ok(())
}
