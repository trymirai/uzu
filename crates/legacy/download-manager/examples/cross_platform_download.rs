use std::path::PathBuf;

use download_manager::{DownloadManager, DownloadPhase, DownloadTaskRequest};
use kiban::rt::RuntimeHandle;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manager = <dyn DownloadManager>::system_default(RuntimeHandle::current()).await?;
    let request = DownloadTaskRequest::file()
        .destination(PathBuf::from("/tmp/test_tokenizer.json"))
        .source_url("https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json")
        .build();
    let task = manager.download_task(request).await?;
    task.delete().await?;

    let mut progress = task.progress();
    task.download().await?;
    while let Some(Ok(state)) = progress.next().await {
        println!("Progress: {} / {} bytes ({:?})", state.downloaded_bytes, state.total_bytes, state.phase);
        if matches!(state.phase, DownloadPhase::Downloaded {} | DownloadPhase::Error { .. }) {
            break;
        }
    }
    println!("Download finished!");
    Ok(())
}
