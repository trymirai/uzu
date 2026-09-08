use std::{sync::Arc, time::Duration};

use futures_util::StreamExt;
use kiban::{fs, fs::PartFile, time::Instant};
use reqwest::{
    Client, StatusCode,
    header::{CONTENT_LENGTH, CONTENT_RANGE, RANGE},
};
use tokio::sync::{oneshot::Sender as TokioOneshotSender, watch::Receiver as TokioWatchReceiver};

use crate::{
    backends::{
        BackendEvent, BackendEventSender, DownloadGeneration,
        universal::{ContentRange, UniversalBackendError},
    },
    file_download::DownloadConfig,
};

const RETRIES: u16 = 3;
const RETRY_DELAY: Duration = Duration::from_millis(250);
const PROGRESS_INTERVAL: Duration = Duration::from_millis(500);

pub struct UniversalStream {
    client: Client,
    config: Arc<DownloadConfig>,
    generation: DownloadGeneration,
    events: BackendEventSender,
    pause: TokioWatchReceiver<bool>,
    completion: TokioOneshotSender<()>,
}

impl UniversalStream {
    pub fn new(
        config: Arc<DownloadConfig>,
        generation: DownloadGeneration,
        events: BackendEventSender,
        pause: TokioWatchReceiver<bool>,
        completion: TokioOneshotSender<()>,
    ) -> Self {
        Self {
            client: Client::new(),
            config,
            generation,
            events,
            pause,
            completion,
        }
    }

    pub async fn run(mut self) {
        let mut attempt = 0_u16;
        let outcome = loop {
            match self.attempt().await {
                Err(error @ (UniversalBackendError::Http(_) | UniversalBackendError::Io(_))) if attempt < RETRIES => {
                    attempt += 1;
                    tracing::debug!(%error, attempt, "retrying universal download");
                    kiban::time::sleep(RETRY_DELAY).await;
                },
                outcome => break outcome,
            }
        };
        match outcome {
            Ok(true) => {},
            Ok(false) => {
                self.events
                    .send_terminal(BackendEvent::Completed {
                        generation: self.generation,
                    })
                    .await
            },
            Err(error) => {
                self.events
                    .send_terminal(BackendEvent::Error {
                        generation: self.generation,
                        message: error.to_string(),
                    })
                    .await
            },
        }
        let _ = self.completion.send(());
    }

    async fn attempt(&mut self) -> Result<bool, UniversalBackendError> {
        let config = &self.config;
        let artifact = config.resume_artifact_path.as_path();
        let mut resume_from = fs::asyn::file_length(artifact).await.unwrap_or(0);
        let mut request = self.client.get(&config.source_url);
        if resume_from > 0 {
            request = request.header(RANGE, format!("bytes={resume_from}-"));
        }
        let response = tokio::select! {
            _ = Self::paused(&mut self.pause) => return Ok(true),
            response = request.send() => response?,
        };
        let status = response.status();
        if resume_from > 0 {
            match status {
                StatusCode::PARTIAL_CONTENT => {
                    let range = ContentRange::parse(response.headers().get(CONTENT_RANGE))?;
                    if range.start != Some(resume_from) {
                        return Err(UniversalBackendError::Protocol(format!(
                            "server returned bytes starting at {} but client requested {resume_from}",
                            range.start.unwrap_or(0)
                        )));
                    }
                },
                StatusCode::OK => resume_from = 0,
                StatusCode::RANGE_NOT_SATISFIABLE => {
                    let total =
                        ContentRange::parse(response.headers().get(CONTENT_RANGE)).ok().and_then(|range| range.total);
                    if total == Some(resume_from) {
                        self.events.send_progress(self.generation, resume_from, Some(resume_from));
                        fs::asyn::rename(artifact, &config.destination).await?;
                        return Ok(false);
                    }
                    let _ = fs::asyn::remove_file(artifact).await;
                    return Err(UniversalBackendError::Protocol(format!(
                        "server did not honor range request: status {status}"
                    )));
                },
                _ => {
                    response.error_for_status_ref()?;
                    return Err(UniversalBackendError::Protocol(format!(
                        "server did not honor range request: status {status}"
                    )));
                },
            }
        }
        let response = response.error_for_status()?;
        let total_bytes = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
            .map(|remaining| remaining.saturating_add(resume_from))
            .or(config.expected_bytes);

        let mut file = <dyn PartFile>::new(artifact, resume_from).await?;
        let mut downloaded_bytes = resume_from;
        let mut last_progress = Instant::now().checked_sub(PROGRESS_INTERVAL).unwrap_or_else(Instant::now);
        let mut body = response.bytes_stream();
        loop {
            let chunk = tokio::select! {
                _ = Self::paused(&mut self.pause) => {
                    file.flush().await?;
                    self.events.send_progress(self.generation, downloaded_bytes, total_bytes);
                    return Ok(true);
                },
                chunk = body.next() => chunk,
            };
            let Some(chunk) = chunk else {
                break;
            };
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded_bytes += chunk.len() as u64;
            if last_progress.elapsed() >= PROGRESS_INTERVAL {
                self.events.send_progress(self.generation, downloaded_bytes, total_bytes);
                last_progress = Instant::now();
            }
        }
        file.flush().await?;
        self.events.send_progress(self.generation, downloaded_bytes, total_bytes.or(Some(downloaded_bytes)));
        fs::asyn::rename(artifact, &config.destination).await?;
        Ok(false)
    }

    async fn paused(pause: &mut TokioWatchReceiver<bool>) {
        if *pause.borrow() {
            return;
        }
        let _ = pause.changed().await;
    }
}
