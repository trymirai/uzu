use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use futures_util::StreamExt;
use http::StatusCode;
#[cfg(not(target_family = "wasm"))]
use http::uri::Scheme;
use kiban::{fs, fs::PartFile, rt::RuntimeHandle, time::Instant};
use reqwest::{
    Client,
    header::{CONTENT_RANGE, RANGE},
};
#[cfg(not(target_family = "wasm"))]
use reqwest::{header::ACCEPT_ENCODING, redirect::Policy};
use tokio::sync::{
    oneshot::{Sender as TokioOneshotSender, channel as tokio_oneshot_channel},
    watch::{Receiver as TokioWatchReceiver, channel as tokio_watch_channel},
};

use crate::{
    backends::universal::{UniversalActiveTask, UniversalBackend, UniversalBackendError},
    file_download_task_actor::BackendEvent,
    lock_manager::DestinationLockLease,
    traits::{ActiveDownloadGeneration, BackendContext, BackendEventSender, DownloadConfig},
};

#[derive(Clone, Debug)]
pub struct UniversalBackendContext {
    runtime_handle: RuntimeHandle,
    pub retries: u16,
    pub progress_interval_ms: u64,
}

impl UniversalBackendContext {
    pub fn new(runtime_handle: RuntimeHandle) -> Self {
        Self {
            runtime_handle,
            retries: 3,
            progress_interval_ms: 500,
        }
    }
}

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
impl BackendContext for UniversalBackendContext {
    type Backend = UniversalBackend;

    async fn download(
        &self,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        backend_event_sender: BackendEventSender,
        _destination_lease: &DestinationLockLease,
    ) -> Result<UniversalActiveTask, UniversalBackendError> {
        let resume_artifact_path = config.destination.with_added_extension("part");
        self.start(config, generation, resume_artifact_path, backend_event_sender).await
    }

    async fn resume(
        &self,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        resume_artifact_path: &Path,
        backend_event_sender: BackendEventSender,
        _destination_lease: &DestinationLockLease,
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
            fs::asyn::create_dir_all(parent).await.map_err(|error| UniversalBackendError::Io(error.to_string()))?;
        }

        let retry_count = self.retries;
        let progress_interval = Duration::from_millis(self.progress_interval_ms);
        let (pause_sender, pause_receiver) = tokio_watch_channel(false);
        let (completion_sender, completion_receiver) = tokio_oneshot_channel();
        let task_handle = self.runtime_handle.spawn(download_streaming(
            config,
            generation,
            resume_artifact_path.clone(),
            backend_event_sender,
            retry_count,
            progress_interval,
            pause_receiver,
            completion_sender,
        ));

        Ok(UniversalActiveTask::new(Box::from([task_handle]), pause_sender, completion_receiver, resume_artifact_path))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DownloadStreamCompletion {
    Completed,
    Paused,
}

async fn download_streaming(
    config: Arc<DownloadConfig>,
    generation: ActiveDownloadGeneration,
    resume_artifact_path: PathBuf,
    backend_event_sender: BackendEventSender,
    retry_count: u16,
    progress_interval: Duration,
    pause_receiver: TokioWatchReceiver<bool>,
    completion_sender: TokioOneshotSender<()>,
) {
    let mut pause_receiver = pause_receiver;
    let result = download_streaming_with_retries(
        Arc::clone(&config),
        generation,
        &resume_artifact_path,
        &backend_event_sender,
        retry_count,
        progress_interval,
        &mut pause_receiver,
    )
    .await;

    match result {
        Ok(DownloadStreamCompletion::Completed) => {
            let _ = backend_event_sender.send_terminal(BackendEvent::completed(generation)).await;
        },
        Ok(DownloadStreamCompletion::Paused) => {},
        Err(error) => {
            let _ = backend_event_sender.send_terminal(BackendEvent::error(generation, error)).await;
        },
    }
    let _ = completion_sender.send(());
}

async fn download_streaming_with_retries(
    config: Arc<DownloadConfig>,
    generation: ActiveDownloadGeneration,
    resume_artifact_path: &Path,
    backend_event_sender: &BackendEventSender,
    retry_count: u16,
    progress_interval: Duration,
    pause_receiver: &mut TokioWatchReceiver<bool>,
) -> Result<DownloadStreamCompletion, String> {
    let client = Client::builder();
    #[cfg(not(target_family = "wasm"))]
    let client = client.redirect(Policy::custom(|attempt| {
        let current_scheme = attempt.previous().last().and_then(|url| Scheme::try_from(url.scheme()).ok());
        let next_scheme = Scheme::try_from(attempt.url().scheme()).ok();
        let downgrade = matches!(
            (current_scheme, next_scheme),
            (Some(current), Some(next)) if current == Scheme::HTTPS && next != Scheme::HTTPS
        );
        if downgrade || attempt.previous().len() >= 10 {
            attempt.error("unsafe redirect")
        } else {
            attempt.follow()
        }
    }));
    let client = client.build().map_err(|error| error.to_string())?;
    let mut attempt = 0_u16;
    loop {
        match download_once(
            &client,
            Arc::clone(&config),
            generation,
            resume_artifact_path,
            backend_event_sender,
            progress_interval,
            pause_receiver,
        )
        .await
        {
            Ok(completion) => return Ok(completion),
            Err(error) if attempt < retry_count => {
                attempt = attempt.saturating_add(1);
                kiban::time::sleep(Duration::from_millis(250)).await;
                tracing::debug!("retrying universal download after error: {error}");
            },
            Err(error) => return Err(error),
        }
    }
}

async fn download_once(
    client: &Client,
    config: Arc<DownloadConfig>,
    generation: ActiveDownloadGeneration,
    resume_artifact_path: &Path,
    backend_event_sender: &BackendEventSender,
    progress_interval: Duration,
    pause_receiver: &mut TokioWatchReceiver<bool>,
) -> Result<DownloadStreamCompletion, String> {
    let resume_from_bytes = fs::asyn::file_length(resume_artifact_path).await.unwrap_or(0);
    let request = config.request.apply(client.get(&config.request.url)).map_err(|error| error.to_string())?;
    #[cfg(not(target_family = "wasm"))]
    let request = request.header(ACCEPT_ENCODING, "identity");
    let mut request = request;
    if resume_from_bytes > 0 {
        request = request.header(RANGE, format!("bytes={resume_from_bytes}-"));
    }

    let response = tokio::select! {
        _ = wait_for_pause(pause_receiver) => return Ok(DownloadStreamCompletion::Paused),
        response = request.send() => response.map_err(|error| error.without_url().to_string())?,
    };
    let status = response.status();
    let content_range = response.headers().get(CONTENT_RANGE).cloned();
    let resume_from_bytes = if resume_from_bytes > 0 {
        match status {
            StatusCode::PARTIAL_CONTENT => {
                let validation = content_range
                    .as_ref()
                    .ok_or_else(|| "server returned 206 without Content-Range header".to_string())
                    .and_then(|header| {
                        header.to_str().map_err(|error| format!("non-utf8 Content-Range header: {error}"))
                    })
                    .and_then(|header| validate_content_range(header, resume_from_bytes, config.expected_bytes));
                if let Err(message) = validation {
                    return Err(discard_invalid_resume(resume_artifact_path, message).await);
                }
                resume_from_bytes
            },
            StatusCode::OK => 0,
            StatusCode::RANGE_NOT_SATISFIABLE => {
                let advertised_total =
                    content_range.as_ref().and_then(|header| header.to_str().ok()).and_then(parse_content_range_total);
                if advertised_total == Some(resume_from_bytes) {
                    backend_event_sender.send_progress(generation, resume_from_bytes, Some(resume_from_bytes)).await;
                    return fs::asyn::rename(resume_artifact_path, &config.destination)
                        .await
                        .map(|()| DownloadStreamCompletion::Completed)
                        .map_err(|error| error.to_string());
                }
                let _ = fs::asyn::remove_file(resume_artifact_path).await;
                return Err(format!("server did not honor range request: status {status}"));
            },
            _ => return Err(format!("server did not honor range request: status {status}")),
        }
    } else {
        resume_from_bytes
    };
    let response = response.error_for_status().map_err(|error| error.without_url().to_string())?;
    let remaining_bytes = response.content_length();
    let total_bytes = remaining_bytes
        .map(|remaining_bytes| remaining_bytes.saturating_add(resume_from_bytes))
        .or(config.expected_bytes);

    let mut file =
        <dyn PartFile>::new(resume_artifact_path, resume_from_bytes).await.map_err(|error| error.to_string())?;
    let mut downloaded_bytes = resume_from_bytes;
    let mut last_progress_emit = Instant::now().checked_sub(progress_interval).unwrap_or_else(Instant::now);
    let mut stream = response.bytes_stream();

    loop {
        let chunk = tokio::select! {
            _ = wait_for_pause(pause_receiver) => {
                file.flush().await.map_err(|error| error.to_string())?;
                backend_event_sender.send_progress(generation, downloaded_bytes, total_bytes).await;
                return Ok(DownloadStreamCompletion::Paused);
            },
            chunk = stream.next() => chunk,
        };

        let Some(chunk) = chunk else {
            break;
        };
        let chunk = chunk.map_err(|error| error.without_url().to_string())?;
        let chunk_len = u64::try_from(chunk.len()).map_err(|_| "response chunk length overflow".to_string())?;
        let next_downloaded_bytes =
            downloaded_bytes.checked_add(chunk_len).ok_or_else(|| "downloaded byte count overflow".to_string())?;
        if let Some(expected_bytes) = config.expected_bytes
            && next_downloaded_bytes > expected_bytes
        {
            return Err(format!("response exceeded the registry size of {expected_bytes} bytes"));
        }
        file.write_all(&chunk).await.map_err(|error| error.to_string())?;
        downloaded_bytes = next_downloaded_bytes;

        if last_progress_emit.elapsed() >= progress_interval {
            backend_event_sender.send_progress(generation, downloaded_bytes, total_bytes).await;
            last_progress_emit = Instant::now();
        }
    }

    file.flush().await.map_err(|error| error.to_string())?;
    backend_event_sender.send_progress(generation, downloaded_bytes, total_bytes.or(Some(downloaded_bytes))).await;
    fs::asyn::rename(resume_artifact_path, &config.destination)
        .await
        .map(|()| DownloadStreamCompletion::Completed)
        .map_err(|error| error.to_string())
}

async fn wait_for_pause(pause_receiver: &mut TokioWatchReceiver<bool>) {
    if *pause_receiver.borrow() {
        return;
    }
    let _ = pause_receiver.changed().await;
}

async fn discard_invalid_resume(
    resume_artifact_path: &Path,
    message: impl Into<String>,
) -> String {
    let _ = fs::asyn::remove_file(resume_artifact_path).await;
    message.into()
}

fn validate_content_range(
    header: &str,
    requested_start: u64,
    expected_total: Option<u64>,
) -> Result<(), String> {
    let value = header.strip_prefix("bytes ").ok_or_else(|| format!("malformed Content-Range: {header}"))?;
    let (range, _) = value.split_once('/').ok_or_else(|| format!("malformed Content-Range: {header}"))?;
    let (start, _) = range.split_once('-').ok_or_else(|| format!("malformed Content-Range: {header}"))?;
    let start = start.parse::<u64>().map_err(|_| format!("malformed Content-Range: {header}"))?;
    if start != requested_start {
        return Err(format!("server returned bytes starting at {start} but client requested {requested_start}"));
    }
    if let Some(expected) = expected_total {
        let total = parse_content_range_total(header)
            .ok_or_else(|| format!("server returned Content-Range without a total: {header}"))?;
        if total != expected {
            return Err(format!("server advertised {total} bytes but registry declared {expected}"));
        }
    }
    Ok(())
}

fn parse_content_range_total(header_value: &str) -> Option<u64> {
    let value = header_value.strip_prefix("bytes ")?.trim_start();
    let (_, total) = value.split_once('/')?;
    if total == "*" {
        return None;
    }
    total.parse::<u64>().ok()
}
