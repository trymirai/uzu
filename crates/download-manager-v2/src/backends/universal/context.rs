use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use async_fetcher::{FetchEvent, Fetcher, Source};
use async_shutdown::ShutdownManager;
use futures_util::{StreamExt, stream};
use tokio::sync::mpsc::unbounded_channel as tokio_unbounded_channel;
use tokio_stream::wrappers::UnboundedReceiverStream;

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
            connections_per_file: 4,
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

        let resume_from_bytes =
            tokio::fs::metadata(&resume_artifact_path).await.ok().map(|metadata| metadata.len()).unwrap_or(0);
        let shutdown_manager = ShutdownManager::new();
        let (fetch_event_sender, fetch_event_receiver) = tokio_unbounded_channel();

        let fetcher = Fetcher::<()>::default()
            .connections_per_file(self.connections_per_file)
            .retries(self.retries)
            .progress_interval(self.progress_interval_ms)
            .shutdown(shutdown_manager.clone())
            .events(fetch_event_sender)
            .build();
        let source = Source::builder(Arc::from(config.destination.clone()), config.source_url.clone().into())
            .partial(Arc::from(resume_artifact_path.clone()))
            .build();

        let event_task = spawn_fetch_event_task(
            fetch_event_receiver,
            generation,
            resume_from_bytes,
            config.expected_bytes,
            backend_event_sender.clone(),
        );
        let fetch_task = spawn_fetch_task(fetcher, source, generation, backend_event_sender);

        Ok(UniversalActiveTask::new(shutdown_manager, Box::from([event_task, fetch_task]), resume_artifact_path))
    }
}

fn spawn_fetch_event_task(
    fetch_event_receiver: tokio::sync::mpsc::UnboundedReceiver<(Arc<Path>, Arc<()>, FetchEvent)>,
    generation: ActiveDownloadGeneration,
    resume_from_bytes: u64,
    expected_bytes: Option<u64>,
    backend_event_sender: BackendEventSender,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut total_size = None;
        let mut downloaded_bytes = resume_from_bytes;
        let mut fetch_event_stream = UnboundedReceiverStream::new(fetch_event_receiver);

        while let Some((_path, _data, fetch_event)) = fetch_event_stream.next().await {
            match fetch_event {
                FetchEvent::ContentLength(length) => {
                    total_size = Some(length.saturating_add(resume_from_bytes));
                },
                FetchEvent::Progress(bytes) => {
                    downloaded_bytes = downloaded_bytes.saturating_add(bytes);
                    let total_bytes = total_size.or(expected_bytes).unwrap_or(downloaded_bytes);
                    backend_event_sender.send_progress(generation, downloaded_bytes, Some(total_bytes)).await;
                },
                FetchEvent::Fetched => {
                    let _ = backend_event_sender.send_terminal(BackendEvent::completed(generation)).await;
                    break;
                },
                FetchEvent::Fetching | FetchEvent::Retrying => {},
            }
        }
    })
}

fn spawn_fetch_task(
    fetcher: Arc<Fetcher<()>>,
    source: Source,
    generation: ActiveDownloadGeneration,
    backend_event_sender: BackendEventSender,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let source_stream = stream::once(async { (source, Arc::new(())) });
        let fetch_task = fetcher.stream_from(source_stream, 1);
        futures_util::pin_mut!(fetch_task);

        while let Some((_path, _data, result)) = fetch_task.next().await {
            if let Err(error) = result {
                let _ = backend_event_sender.send_terminal(BackendEvent::error(generation, error.to_string())).await;
            }
        }
    })
}
