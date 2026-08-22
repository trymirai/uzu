use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use futures_util::StreamExt;
use kiban::{fs, fs::PartFile, rt::RuntimeHandle, time::Instant};
use reqwest::{
    StatusCode,
    header::{CONTENT_LENGTH, CONTENT_RANGE, HeaderValue, RANGE},
    redirect::Policy,
};
use tokio::sync::{
    oneshot::channel as tokio_oneshot_channel,
    watch::{Receiver as TokioWatchReceiver, channel as tokio_watch_channel},
};

use crate::{
    DownloadError,
    backends::{
        common::{ensure_owned_directory, reject_symlink_components},
        universal::{UniversalActiveTask, UniversalBackend, UniversalBackendError},
    },
    file_download_task_actor::BackendEvent,
    lock_manager::DestinationLockLease,
    recovery_metadata::{prepare_fresh_recovery, prepare_resume_recovery},
    traits::{ActiveDownloadGeneration, ActiveTaskPauseOutcome, BackendContext, BackendEventSender, DownloadConfig},
};

#[derive(Clone, Debug)]
pub struct UniversalBackendContext {
    runtime_handle: RuntimeHandle,
    client: reqwest::Client,
    pub retries: u16,
    pub progress_interval_ms: u64,
}

impl UniversalBackendContext {
    pub fn new(runtime_handle: RuntimeHandle) -> Self {
        Self {
            runtime_handle,
            client: reqwest::Client::builder()
                .redirect(safe_redirect_policy())
                .build()
                .expect("the static HTTP client configuration must be valid"),
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
        destination_lease: &DestinationLockLease,
    ) -> Result<UniversalActiveTask, UniversalBackendError> {
        let resume_artifact_path = config.resume_artifact_path("part");
        ensure_owned_directory(&config.artifact_root).await?;
        prepare_fresh_recovery(&config, &resume_artifact_path, destination_lease).await?;
        self.start(config, generation, resume_artifact_path, backend_event_sender).await
    }

    async fn resume(
        &self,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        resume_artifact_path: &Path,
        backend_event_sender: BackendEventSender,
        destination_lease: &DestinationLockLease,
    ) -> Result<UniversalActiveTask, UniversalBackendError> {
        ensure_owned_directory(&config.artifact_root).await?;
        prepare_resume_recovery(&config, resume_artifact_path, destination_lease).await?;
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
            reject_symlink_components(parent).await.map_err(|error| UniversalBackendError::Io(error.to_string()))?;
            fs::asyn::create_dir_all(parent).await.map_err(|error| UniversalBackendError::Io(error.to_string()))?;
            reject_symlink_components(parent).await.map_err(|error| UniversalBackendError::Io(error.to_string()))?;
        }

        let retry_count = self.retries;
        let client = self.client.clone();
        let progress_interval = Duration::from_millis(self.progress_interval_ms);
        let (pause_sender, pause_receiver) = tokio_watch_channel(false);
        let (completion_sender, completion_receiver) = tokio_oneshot_channel();
        let task_handle = self.runtime_handle.spawn(download_streaming(
            client,
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
    client: reqwest::Client,
    config: Arc<DownloadConfig>,
    generation: ActiveDownloadGeneration,
    resume_artifact_path: PathBuf,
    backend_event_sender: BackendEventSender,
    retry_count: u16,
    progress_interval: Duration,
    pause_receiver: TokioWatchReceiver<bool>,
    completion_sender: tokio::sync::oneshot::Sender<ActiveTaskPauseOutcome>,
) {
    let mut pause_receiver = pause_receiver;
    let result = download_streaming_with_retries(
        &client,
        Arc::clone(&config),
        generation,
        &resume_artifact_path,
        &backend_event_sender,
        retry_count,
        progress_interval,
        &mut pause_receiver,
    )
    .await;

    let pause_outcome = match result {
        Ok(DownloadStreamCompletion::Completed) => {
            let _ = backend_event_sender.send_terminal(BackendEvent::completed(generation)).await;
            ActiveTaskPauseOutcome::Completed
        },
        Ok(DownloadStreamCompletion::Paused) => ActiveTaskPauseOutcome::Paused(resume_artifact_path),
        Err(error) => {
            let _ = backend_event_sender.send_terminal(BackendEvent::error(generation, error.clone())).await;
            ActiveTaskPauseOutcome::Failed(error)
        },
    };
    let _ = completion_sender.send(pause_outcome);
}

async fn download_streaming_with_retries(
    client: &reqwest::Client,
    config: Arc<DownloadConfig>,
    generation: ActiveDownloadGeneration,
    resume_artifact_path: &Path,
    backend_event_sender: &BackendEventSender,
    retry_count: u16,
    progress_interval: Duration,
    pause_receiver: &mut TokioWatchReceiver<bool>,
) -> Result<DownloadStreamCompletion, DownloadError> {
    let mut attempt = 0_u16;
    loop {
        match download_once(
            client,
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
            Err(error) if error.is_retryable_transfer_failure() && attempt < retry_count => {
                attempt = attempt.saturating_add(1);
                kiban::time::sleep(Duration::from_millis(250)).await;
                tracing::debug!("retrying universal download after error: {error}");
            },
            Err(error) => return Err(error),
        }
    }
}

async fn download_once(
    client: &reqwest::Client,
    config: Arc<DownloadConfig>,
    generation: ActiveDownloadGeneration,
    resume_artifact_path: &Path,
    backend_event_sender: &BackendEventSender,
    progress_interval: Duration,
    pause_receiver: &mut TokioWatchReceiver<bool>,
) -> Result<DownloadStreamCompletion, DownloadError> {
    let resume_from_bytes = fs::asyn::file_length(resume_artifact_path).await.unwrap_or(0);
    let mut headers = config.request.headers.as_header_map().clone();
    headers.remove(RANGE);
    if resume_from_bytes > 0 {
        let range = HeaderValue::from_str(&format!("bytes={resume_from_bytes}-"))
            .map_err(|_| DownloadError::Protocol("invalid byte range".to_string()))?;
        headers.insert(RANGE, range);
    }
    let request = client.get(&config.request.url).headers(headers);

    let response = tokio::select! {
        _ = wait_for_pause(pause_receiver) => return Ok(DownloadStreamCompletion::Paused),
        response = request.send() => response.map_err(classify_reqwest_error)?,
    };
    let status = response.status();
    let content_range = response.headers().get(CONTENT_RANGE).cloned();
    let resume_from_bytes = if resume_from_bytes > 0 {
        if status == StatusCode::RANGE_NOT_SATISFIABLE {
            let advertised_total =
                content_range.as_ref().and_then(|header| header.to_str().ok()).and_then(parse_content_range_total);
            if advertised_total == Some(resume_from_bytes) {
                backend_event_sender.send_progress(generation, resume_from_bytes, Some(resume_from_bytes)).await;
                reject_symlink_components(&config.destination)
                    .await
                    .map_err(|error| DownloadError::Io(error.to_string()))?;
                return fs::asyn::rename(resume_artifact_path, &config.destination)
                    .await
                    .map(|()| DownloadStreamCompletion::Completed)
                    .map_err(|error| DownloadError::Io(error.to_string()));
            }
        }
        reject_http_error(status)?;
        match status {
            StatusCode::PARTIAL_CONTENT => {
                let header = content_range.as_ref().ok_or_else(|| {
                    DownloadError::Protocol("server returned 206 without Content-Range header".to_string())
                })?;
                let header_value = header
                    .to_str()
                    .map_err(|error| DownloadError::Protocol(format!("non-utf8 Content-Range header: {error}")))?;
                let advertised_start = parse_content_range_start(header_value).ok_or_else(|| {
                    DownloadError::Protocol(format!("server returned malformed Content-Range header: {header_value}"))
                })?;
                if advertised_start != resume_from_bytes {
                    return Err(DownloadError::Protocol(format!(
                        "server returned bytes starting at {advertised_start} but client requested {resume_from_bytes}"
                    )));
                }
                resume_from_bytes
            },
            StatusCode::OK => 0,
            _ => {
                return Err(DownloadError::Protocol(format!("server did not honor range request: status {status}")));
            },
        }
    } else {
        reject_http_error(status)?;
        resume_from_bytes
    };
    let remaining_bytes = response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let total_bytes = remaining_bytes
        .map(|remaining_bytes| remaining_bytes.saturating_add(resume_from_bytes))
        .or(config.expected_bytes);

    let mut file = <dyn PartFile>::new(resume_artifact_path, resume_from_bytes)
        .await
        .map_err(|error| DownloadError::Io(error.to_string()))?;
    let mut downloaded_bytes = resume_from_bytes;
    let mut last_progress_emit = Instant::now().checked_sub(progress_interval).unwrap_or_else(Instant::now);
    let mut stream = response.bytes_stream();

    loop {
        let chunk = tokio::select! {
            _ = wait_for_pause(pause_receiver) => {
                file.flush().await.map_err(|error| DownloadError::Io(error.to_string()))?;
                backend_event_sender.send_progress(generation, downloaded_bytes, total_bytes).await;
                return Ok(DownloadStreamCompletion::Paused);
            },
            chunk = stream.next() => chunk,
        };

        let Some(chunk) = chunk else {
            break;
        };
        let chunk = chunk.map_err(|error| DownloadError::Transport(error.to_string()))?;
        file.write_all(&chunk).await.map_err(|error| DownloadError::Io(error.to_string()))?;
        downloaded_bytes = downloaded_bytes.saturating_add(chunk.len() as u64);

        if last_progress_emit.elapsed() >= progress_interval {
            backend_event_sender.send_progress(generation, downloaded_bytes, total_bytes).await;
            last_progress_emit = Instant::now();
        }
    }

    file.flush().await.map_err(|error| DownloadError::Io(error.to_string()))?;
    backend_event_sender.send_progress(generation, downloaded_bytes, total_bytes).await;
    reject_symlink_components(&config.destination).await.map_err(|error| DownloadError::Io(error.to_string()))?;
    fs::asyn::rename(resume_artifact_path, &config.destination)
        .await
        .map(|()| DownloadStreamCompletion::Completed)
        .map_err(|error| DownloadError::Io(error.to_string()))
}

fn safe_redirect_policy() -> Policy {
    Policy::custom(|attempt| {
        if attempt.previous().len() > 10 {
            return attempt.error("too many redirects");
        }

        if is_https_downgrade(attempt.previous().last(), attempt.url()) {
            attempt.error("refusing an HTTPS-to-HTTP download redirect")
        } else {
            attempt.follow()
        }
    })
}

fn is_https_downgrade(
    previous: Option<&reqwest::Url>,
    next: &reqwest::Url,
) -> bool {
    previous.is_some_and(|previous| previous.scheme() == "https" && next.scheme() != "https")
}

fn classify_reqwest_error(error: reqwest::Error) -> DownloadError {
    if error.is_redirect() {
        DownloadError::Protocol(error.to_string())
    } else {
        DownloadError::Transport(error.to_string())
    }
}

fn reject_http_error(status: StatusCode) -> Result<(), DownloadError> {
    if !status.is_success() {
        Err(DownloadError::from_http_status(status.as_u16()))
    } else {
        Ok(())
    }
}

async fn wait_for_pause(pause_receiver: &mut TokioWatchReceiver<bool>) {
    if *pause_receiver.borrow() {
        return;
    }
    let _ = pause_receiver.changed().await;
}

fn parse_content_range_start(header_value: &str) -> Option<u64> {
    let value = header_value.strip_prefix("bytes ")?.trim_start();
    let (range, _) = value.split_once('/')?;
    let (start, _) = range.split_once('-')?;
    start.parse::<u64>().ok()
}

fn parse_content_range_total(header_value: &str) -> Option<u64> {
    let value = header_value.strip_prefix("bytes ")?.trim_start();
    let (_, total) = value.split_once('/')?;
    if total == "*" {
        return None;
    }
    total.parse::<u64>().ok()
}

#[cfg(test)]
#[path = "../../../tests/unit/backends/universal/context_test.rs"]
mod tests;
