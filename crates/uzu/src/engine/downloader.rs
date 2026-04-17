use nagare::{
    helpers::SharedAccess,
    storage::{
        Error as StorageError, Storage,
        types::{DownloadPhase, DownloadState},
    },
};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

use crate::engine::Error;

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

    pub async fn state(&self) -> Option<DownloadState> {
        self.storage.lock().await.state(&self.identifier).await
    }

    pub async fn resume(&self) -> Result<(), Error> {
        let state = self.state().await.ok_or(StorageError::ItemNotFound {
            identifier: self.identifier.clone(),
        })?;
        let result = match state.phase {
            DownloadPhase::Downloading | DownloadPhase::Downloaded => Ok(()),
            DownloadPhase::NotDownloaded | DownloadPhase::Paused | DownloadPhase::Locked => {
                Ok(self.storage.lock().await.download(&self.identifier).await?)
            },
            DownloadPhase::Error(_) => {
                let storage = self.storage.lock().await;
                storage.delete(&self.identifier).await?;
                Ok(storage.download(&self.identifier).await?)
            },
        };
        result
    }

    pub async fn pause(&self) -> Result<(), Error> {
        Ok(self.storage.lock().await.pause(&self.identifier).await?)
    }

    pub async fn delete(&self) -> Result<(), Error> {
        Ok(self.storage.lock().await.delete(&self.identifier).await?)
    }

    pub async fn progress(&self) -> Option<Stream> {
        let identifier = self.identifier.clone();
        let Some(state) = self.state().await else {
            return None;
        };
        if matches!(state.phase, DownloadPhase::Downloaded) {
            return None;
        }
        let stream = self.storage.lock().await.subscribe();
        Some(Stream::new(identifier, stream))
    }
}

pub struct Stream {
    identifier: String,
    stream: SharedAccess<Option<BroadcastStream<(String, DownloadState)>>>,
}

impl Stream {
    pub(crate) fn new(
        identifier: String,
        stream: BroadcastStream<(String, DownloadState)>,
    ) -> Self {
        Self {
            identifier,
            stream: SharedAccess::new(Some(stream)),
        }
    }

    pub async fn next(&self) -> Option<Update> {
        let mut stream_guard = self.stream.lock().await;
        let stream = stream_guard.as_mut()?;
        while let Some(result) = stream.next().await {
            if let Ok((identifier, state)) = result {
                if identifier == self.identifier {
                    let update = Update {
                        bytes_total: state.total_bytes,
                        bytes_downloaded: state.downloaded_bytes,
                    };
                    match state.phase {
                        DownloadPhase::Downloading | DownloadPhase::Locked => {},
                        DownloadPhase::NotDownloaded
                        | DownloadPhase::Downloaded
                        | DownloadPhase::Error(_)
                        | DownloadPhase::Paused => {
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

pub struct Update {
    pub bytes_total: u64,
    pub bytes_downloaded: u64,
}

impl Update {
    pub fn progress(&self) -> f32 {
        if self.bytes_total == 0 {
            0.0
        } else {
            self.bytes_downloaded as f32 / self.bytes_total as f32
        }
    }
}
