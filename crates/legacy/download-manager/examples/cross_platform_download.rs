use std::{env, path::PathBuf};

use download_manager::{DownloadManager, DownloadManagerType, DownloadPhase, DownloadTaskRequest};
use kiban::rt::RuntimeHandle;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let source_url = args
        .next()
        .unwrap_or_else(|| "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json".to_string());
    let destination = args.next().unwrap_or_else(|| "/tmp/test_tokenizer.json".to_string());
    let manager = DownloadManager::new(DownloadManagerType::default(), RuntimeHandle::current());
    let request = DownloadTaskRequest::file().destination(PathBuf::from(destination)).source_url(source_url).build();
    let task = manager.download_task(request).await?;
    println!("Initial state: {:?}", task.state());

    let mut progress = task.progress();
    task.download().await?;
    if !task.state().is_in_progress() {
        println!("Final state: {:?}", task.state());
        return Ok(());
    }
    while let Some(state) = progress.next().await {
        println!("Progress: {} / {} bytes ({:?})", state.downloaded_bytes, state.total_bytes, state.phase);
        if matches!(state.phase, DownloadPhase::Downloaded {} | DownloadPhase::Error { .. }) {
            break;
        }
    }
    println!("Final state: {:?}", task.state());
    Ok(())
}
