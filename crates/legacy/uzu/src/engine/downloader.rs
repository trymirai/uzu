use std::{pin::Pin, sync::Arc};

use serde::{Deserialize, Serialize};
use tokio_stream::{Stream, StreamExt};

use crate::{
    engine::EngineError,
    helpers::SharedAccess,
    storage::{
        Storage, StorageError,
        types::{DownloadPhase, DownloadState},
    },
};

#[bindings::export(Class)]
#[derive(Clone)]
pub struct Downloader {
    identifier: String,
    storage: Arc<Storage>,
}

impl Downloader {
    pub fn new(
        identifier: String,
        storage: Arc<Storage>,
    ) -> Self {
        Self {
            identifier,
            storage,
        }
    }

    fn is_progress_streaming_phase(phase: &DownloadPhase) -> bool {
        matches!(phase, DownloadPhase::Downloading {})
    }
}

#[bindings::export(Implementation)]
impl Downloader {
    #[bindings::export(Method(Getter))]
    pub async fn state(&self) -> Option<DownloadState> {
        self.storage.state(&self.identifier).await
    }

    #[bindings::export(Method)]
    pub async fn resume(&self) -> Result<(), EngineError> {
        let state = self.state().await.ok_or(StorageError::ItemNotFound {
            identifier: self.identifier.clone(),
        })?;

        match state.phase {
            DownloadPhase::Downloading {} | DownloadPhase::Downloaded {} => Ok(()),
            DownloadPhase::NotDownloaded {}
            | DownloadPhase::Paused {}
            | DownloadPhase::Locked {}
            | DownloadPhase::Error {
                ..
            } => self.storage.download(&self.identifier).await.map_err(Into::into),
        }
    }

    #[bindings::export(Method)]
    pub async fn pause(&self) -> Result<(), EngineError> {
        Ok(self.storage.pause(&self.identifier).await?)
    }

    #[bindings::export(Method)]
    pub async fn delete(&self) -> Result<(), EngineError> {
        Ok(self.storage.delete(&self.identifier).await?)
    }

    #[bindings::export(Method)]
    pub async fn progress(&self) -> Result<DownloaderStream, EngineError> {
        let Some(item) = self.storage.get(&self.identifier).await else {
            return Err(EngineError::UnableToGetDownloaderProgressStream {});
        };
        let state = item.state().await;
        if !Self::is_progress_streaming_phase(&state.phase) {
            return Ok(DownloaderStream::empty());
        }
        Ok(DownloaderStream::new(item.watch_states()))
    }
}

#[bindings::export(Class(Stream))]
#[derive(Clone)]
pub struct DownloaderStream {
    stream: SharedAccess<Option<Pin<Box<dyn Stream<Item = DownloadState> + Send>>>>,
}

impl DownloaderStream {
    pub(crate) fn new(stream: impl Stream<Item = DownloadState> + Send + 'static) -> Self {
        Self {
            stream: SharedAccess::new(Some(Box::pin(stream))),
        }
    }

    pub(crate) fn empty() -> Self {
        Self {
            stream: SharedAccess::new(None),
        }
    }
}

#[bindings::export(Implementation)]
impl DownloaderStream {
    #[bindings::export(Method(StreamNext))]
    pub async fn next(&self) -> Option<DownloaderStreamUpdate> {
        let mut stream_guard = self.stream.lock().await;
        let stream = stream_guard.as_mut()?;
        let state = stream.next().await?;
        let update = DownloaderStreamUpdate {
            bytes_total: state.total_bytes,
            bytes_downloaded: state.downloaded_bytes,
        };
        if !matches!(state.phase, DownloadPhase::Downloading {}) {
            *stream_guard = None;
        }
        Some(update)
    }
}

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DownloaderStreamUpdate {
    pub bytes_total: i64,
    pub bytes_downloaded: i64,
}

#[bindings::export(Implementation)]
impl DownloaderStreamUpdate {
    #[bindings::export(Method(Getter))]
    pub fn progress(&self) -> f32 {
        if self.bytes_total == 0 {
            0.0
        } else {
            self.bytes_downloaded as f32 / self.bytes_total as f32
        }
    }
}
