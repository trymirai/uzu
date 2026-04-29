use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use futures_util::StreamExt;
use reqwest::header::{CONTENT_LENGTH, RANGE};
use tokio::{
    fs::{File as TokioFile, OpenOptions as TokioOpenOptions},
    io::AsyncWriteExt,
};

use crate::{
    backends::universal::{UniversalActiveTask, UniversalBackend, UniversalBackendError},
    file_download_task_actor::BackendEvent,
    traits::{ActiveDownloadGeneration, BackendContext, BackendEventSender, DownloadConfig},
};

#[derive(Clone, Debug)]
pub struct UniversalBackendContext {
    pub connections_per_file: u16,
    pub retries: u16,
    pub progress_interval_ms: u64,
}

impl Default for UniversalBackendContext {
    fn default() -> Self {
        Self {
            connections_per_file: 1,
            retries: 3,
            progress_interval_ms: 500,
        }
    }
}

#[async_trait::async_trait]
impl BackendContext for UniversalBackendContext {
    type Backend = UniversalBackend;

    async fn download(
        &self,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        backend_event_sender: BackendEventSender,
    ) -> Result<UniversalActiveTask, UniversalBackendError> {
        let resume_artifact_path = config.destination.with_extension("part");
        self.start(config, generation, resume_artifact_path, backend_event_sender).await
    }

    async fn resume(
        &self,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        resume_artifact_path: &Path,
        backend_event_sender: BackendEventSender,
    ) -> Result<UniversalActiveTask, UniversalBackendError> {
        self.start(config, generation, resume_artifact_path.to_path_buf(), backend_event_sender).await
    }
}

impl UniversalBackendContext {
    async fn start(
        &self,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        resume_artifact_path: PathBuf,
        backend_event_sender: BackendEventSender,
    ) -> Result<UniversalActiveTask, UniversalBackendError> {
        if let Some(parent) = config.destination.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|error| UniversalBackendError::Io(error.to_string()))?;
        }

        let retry_count = self.retries;
        let progress_interval = Duration::from_millis(self.progress_interval_ms);
        let task_handle = tokio::spawn(download_streaming(
            config,
            generation,
            resume_artifact_path.clone(),
            backend_event_sender,
            retry_count,
            progress_interval,
        ));

        Ok(UniversalActiveTask::new(Box::from([task_handle]), resume_artifact_path))
    }
}

async fn download_streaming(
    config: Arc<DownloadConfig>,
    generation: ActiveDownloadGeneration,
    resume_artifact_path: PathBuf,
    backend_event_sender: BackendEventSender,
    retry_count: u16,
    progress_interval: Duration,
) {
    let result = download_streaming_with_retries(
        Arc::clone(&config),
        generation,
        &resume_artifact_path,
        &backend_event_sender,
        retry_count,
        progress_interval,
    )
    .await;

    let terminal_event = match result {
        Ok(()) => BackendEvent::completed(generation),
        Err(error) => BackendEvent::error(generation, error),
    };
    let _ = backend_event_sender.send_terminal(terminal_event).await;
}

async fn download_streaming_with_retries(
    config: Arc<DownloadConfig>,
    generation: ActiveDownloadGeneration,
    resume_artifact_path: &Path,
    backend_event_sender: &BackendEventSender,
    retry_count: u16,
    progress_interval: Duration,
) -> Result<(), String> {
    let mut attempt = 0_u16;
    loop {
        match download_once(
            Arc::clone(&config),
            generation,
            resume_artifact_path,
            backend_event_sender,
            progress_interval,
        )
        .await
        {
            Ok(()) => return Ok(()),
            Err(error) if attempt < retry_count => {
                attempt = attempt.saturating_add(1);
                tokio::time::sleep(Duration::from_millis(250)).await;
                tracing::debug!("retrying universal download after error: {error}");
            },
            Err(error) => return Err(error),
        }
    }
}

async fn download_once(
    config: Arc<DownloadConfig>,
    generation: ActiveDownloadGeneration,
    resume_artifact_path: &Path,
    backend_event_sender: &BackendEventSender,
    progress_interval: Duration,
) -> Result<(), String> {
    let client = reqwest::Client::new();
    let resume_from_bytes =
        tokio::fs::metadata(resume_artifact_path).await.ok().map(|metadata| metadata.len()).unwrap_or(0);
    let mut request = client.get(&config.source_url);
    if resume_from_bytes > 0 {
        request = request.header(RANGE, format!("bytes={resume_from_bytes}-"));
    }

    let response = request.send().await.map_err(|error| error.to_string())?;
    let response = response.error_for_status().map_err(|error| error.to_string())?;
    let remaining_bytes = response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let total_bytes = remaining_bytes
        .map(|remaining_bytes| remaining_bytes.saturating_add(resume_from_bytes))
        .or(config.expected_bytes);

    let mut file = open_part_file(resume_artifact_path, resume_from_bytes).await.map_err(|error| error.to_string())?;
    let mut downloaded_bytes = resume_from_bytes;
    let mut last_progress_emit =
        std::time::Instant::now().checked_sub(progress_interval).unwrap_or_else(std::time::Instant::now);
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| error.to_string())?;
        file.write_all(&chunk).await.map_err(|error| error.to_string())?;
        downloaded_bytes = downloaded_bytes.saturating_add(chunk.len() as u64);

        if last_progress_emit.elapsed() >= progress_interval {
            backend_event_sender.send_progress(generation, downloaded_bytes, total_bytes).await;
            last_progress_emit = std::time::Instant::now();
        }
    }

    file.flush().await.map_err(|error| error.to_string())?;
    backend_event_sender.send_progress(generation, downloaded_bytes, total_bytes.or(Some(downloaded_bytes))).await;
    tokio::fs::rename(resume_artifact_path, &config.destination).await.map_err(|error| error.to_string())
}

async fn open_part_file(
    resume_artifact_path: &Path,
    resume_from_bytes: u64,
) -> Result<TokioFile, std::io::Error> {
    if resume_from_bytes > 0 {
        TokioOpenOptions::new().create(true).append(true).open(resume_artifact_path).await
    } else {
        TokioOpenOptions::new().create(true).write(true).truncate(true).open(resume_artifact_path).await
    }
}
