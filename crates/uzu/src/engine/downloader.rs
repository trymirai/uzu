use serde::{Deserialize, Serialize};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

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
    storage: SharedAccess<Storage>,
}

impl Downloader {
    pub(crate) fn new(
        identifier: String,
        storage: SharedAccess<Storage>,
    ) -> Self {
        Self {
            identifier,
            storage,
        }
    }
}

#[bindings::export(Implementation)]
impl Downloader {
    #[bindings::export(Getter)]
    pub async fn state(&self) -> Option<DownloadState> {
        self.storage.lock().await.state(&self.identifier).await
    }

    #[bindings::export(Method)]
    pub async fn resume(&self) -> Result<(), EngineError> {
        let state = self.state().await.ok_or(StorageError::ItemNotFound {
            identifier: self.identifier.clone(),
        })?;
        let result = match state.phase {
            DownloadPhase::Downloading {} | DownloadPhase::Downloaded {} => Ok(()),
            DownloadPhase::NotDownloaded {} | DownloadPhase::Paused {} | DownloadPhase::Locked {} => {
                Ok(self.storage.lock().await.download(&self.identifier).await?)
            },
            DownloadPhase::Error {
                ..
            } => {
                let storage = self.storage.lock().await;
                storage.delete(&self.identifier).await?;
                Ok(storage.download(&self.identifier).await?)
            },
        };
        result
    }

    #[bindings::export(Method)]
    pub async fn pause(&self) -> Result<(), EngineError> {
        Ok(self.storage.lock().await.pause(&self.identifier).await?)
    }

    #[bindings::export(Method)]
    pub async fn delete(&self) -> Result<(), EngineError> {
        Ok(self.storage.lock().await.delete(&self.identifier).await?)
    }

    #[bindings::export(Method)]
    pub async fn progress(&self) -> Result<DownloaderStream, EngineError> {
        let identifier = self.identifier.clone();
        let Some(state) = self.state().await else {
            return Err(EngineError::UnableToGetDownloaderProgressStream {});
        };
        if matches!(state.phase, DownloadPhase::Downloaded {}) {
            return Err(EngineError::UnableToGetDownloaderProgressStream {});
        }
        let stream = self.storage.lock().await.subscribe();
        Ok(DownloaderStream::new(identifier, stream))
    }
}

#[bindings::export(Stream)]
#[derive(Clone)]
pub struct DownloaderStream {
    identifier: String,
    stream: SharedAccess<Option<BroadcastStream<(String, DownloadState)>>>,
}

impl DownloaderStream {
    pub(crate) fn new(
        identifier: String,
        stream: BroadcastStream<(String, DownloadState)>,
    ) -> Self {
        Self {
            identifier,
            stream: SharedAccess::new(Some(stream)),
        }
    }

    pub(crate) fn empty(identifier: String) -> Self {
        Self {
            identifier,
            stream: SharedAccess::new(None),
        }
    }
}

#[bindings::export(Implementation)]
impl DownloaderStream {
    #[bindings::export(StreamNext)]
    pub async fn next(&self) -> Option<DownloaderStreamUpdate> {
        let mut stream_guard = self.stream.lock().await;
        let stream = stream_guard.as_mut()?;
        while let Some(result) = stream.next().await {
            if let Ok((identifier, state)) = result {
                if identifier == self.identifier {
                    let update = DownloaderStreamUpdate {
                        bytes_total: state.total_bytes,
                        bytes_downloaded: state.downloaded_bytes,
                    };
                    match state.phase {
                        DownloadPhase::Downloading {} | DownloadPhase::Locked {} => {},
                        DownloadPhase::NotDownloaded {}
                        | DownloadPhase::Downloaded {}
                        | DownloadPhase::Error {
                            ..
                        }
                        | DownloadPhase::Paused {} => {
                            *stream_guard = None;
                        },
                    }
                    return Some(update);
                }
            }
        }
        None
    }
}

#[bindings::export(ClassCloneable)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DownloaderStreamUpdate {
    pub bytes_total: i64,
    pub bytes_downloaded: i64,
}

#[bindings::export(Implementation)]
impl DownloaderStreamUpdate {
    #[bindings::export(Getter)]
    pub fn progress(&self) -> f32 {
        if self.bytes_total == 0 {
            0.0
        } else {
            self.bytes_downloaded as f32 / self.bytes_total as f32
        }
    }
}
