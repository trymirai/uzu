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
    pub fn new(
        identifier: String,
        storage: SharedAccess<Storage>,
    ) -> Self {
        Self {
            identifier,
            storage,
        }
    }

    async fn wait_for_resume_to_be_observable(
        &self,
        mut stream: BroadcastStream<(String, DownloadState)>,
    ) -> Result<(), EngineError> {
        if self.state().await.is_some_and(|state| Self::is_resume_observable_phase(&state.phase)) {
            return Ok(());
        }

        while let Some(result) = stream.next().await {
            match result {
                Ok((identifier, state)) if identifier == self.identifier => {
                    if Self::is_resume_observable_phase(&state.phase) {
                        return Ok(());
                    }
                },
                Ok(_) => {},
                Err(error) => {
                    tracing::warn!(
                        identifier = self.identifier,
                        ?error,
                        "downloader resume stream lagged; some updates were dropped"
                    );
                },
            }
        }

        Err(EngineError::UnableToGetDownloaderProgressStream {})
    }

    fn is_progress_streaming_phase(phase: &DownloadPhase) -> bool {
        matches!(phase, DownloadPhase::Downloading {})
    }

    fn is_resume_observable_phase(phase: &DownloadPhase) -> bool {
        matches!(phase, DownloadPhase::Downloading {} | DownloadPhase::Downloaded {} | DownloadPhase::Error { .. })
    }
}

#[bindings::export(Implementation)]
impl Downloader {
    #[bindings::export(Method(Getter))]
    pub async fn state(&self) -> Option<DownloadState> {
        self.storage.lock().await.state(&self.identifier).await
    }

    #[bindings::export(Method)]
    pub async fn resume(&self) -> Result<(), EngineError> {
        let state = self.state().await.ok_or(StorageError::ItemNotFound {
            identifier: self.identifier.clone(),
        })?;

        match state.phase {
            DownloadPhase::Downloading {} | DownloadPhase::Downloaded {} => Ok(()),
            DownloadPhase::NotDownloaded {} | DownloadPhase::Paused {} | DownloadPhase::Locked {} => {
                let stream = self.storage.lock().await.subscribe();
                self.storage.lock().await.download(&self.identifier).await?;
                self.wait_for_resume_to_be_observable(stream).await
            },
            DownloadPhase::Error {
                ..
            } => {
                let stream = self.storage.lock().await.subscribe();
                let storage = self.storage.lock().await;
                storage.delete(&self.identifier).await?;
                storage.download(&self.identifier).await?;
                drop(storage);
                self.wait_for_resume_to_be_observable(stream).await
            },
        }
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
        let Some(state) = self.state().await else {
            return Err(EngineError::UnableToGetDownloaderProgressStream {});
        };
        if !Self::is_progress_streaming_phase(&state.phase) {
            return Ok(DownloaderStream::empty(self.identifier.clone()));
        }
        let stream = self.storage.lock().await.subscribe();
        Ok(DownloaderStream::new(self.identifier.clone(), stream, self.storage.clone()))
    }
}

#[bindings::export(Class(Stream))]
#[derive(Clone)]
pub struct DownloaderStream {
    identifier: String,
    stream: SharedAccess<Option<BroadcastStream<(String, DownloadState)>>>,
    storage: Option<SharedAccess<Storage>>,
}

impl DownloaderStream {
    pub(crate) fn new(
        identifier: String,
        stream: BroadcastStream<(String, DownloadState)>,
        storage: SharedAccess<Storage>,
    ) -> Self {
        Self {
            identifier,
            stream: SharedAccess::new(Some(stream)),
            storage: Some(storage),
        }
    }

    pub(crate) fn empty(identifier: String) -> Self {
        Self {
            identifier,
            stream: SharedAccess::new(None),
            storage: None,
        }
    }

    /// A transient non-`Downloading` phase (for example a momentary `Paused`
    /// while a download is being resumed) is superseded by `Downloading` almost
    /// immediately, whereas a real pause/stop persists. Poll the canonical state
    /// briefly to tell them apart so a blip does not permanently end the stream.
    async fn download_has_settled(&self) -> bool {
        let Some(storage) = self.storage.as_ref() else {
            return true;
        };
        for _ in 0..10 {
            match storage.lock().await.state(&self.identifier).await {
                Some(state) if matches!(state.phase, DownloadPhase::Downloading {} | DownloadPhase::Downloaded {}) => {
                    return false;
                },
                Some(_) => {},
                None => return true,
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        true
    }
}

#[bindings::export(Implementation)]
impl DownloaderStream {
    #[bindings::export(Method(StreamNext))]
    pub async fn next(&self) -> Option<DownloaderStreamUpdate> {
        let mut stream_guard = self.stream.lock().await;
        let stream = stream_guard.as_mut()?;
        while let Some(result) = stream.next().await {
            match result {
                Ok((identifier, state)) => {
                    if identifier == self.identifier {
                        let update = DownloaderStreamUpdate {
                            bytes_total: state.total_bytes,
                            bytes_downloaded: state.downloaded_bytes,
                        };
                        let stream_ended = match state.phase {
                            // Still actively downloading: keep streaming.
                            DownloadPhase::Downloading {} => false,
                            // Terminal: the download is over either way.
                            DownloadPhase::Downloaded {}
                            | DownloadPhase::Error {
                                ..
                            } => true,
                            // Possibly a transient blip while (re)starting. End the
                            // stream only if the model has genuinely stopped, so a
                            // momentary state change does not cut off progress for a
                            // download that is still on its way to completion.
                            DownloadPhase::Paused {} | DownloadPhase::NotDownloaded {} | DownloadPhase::Locked {} => {
                                self.download_has_settled().await
                            },
                        };
                        if stream_ended {
                            *stream_guard = None;
                        }
                        return Some(update);
                    }
                },
                Err(error) => {
                    tracing::warn!(
                        identifier = self.identifier,
                        ?error,
                        "downloader progress stream lagged; some updates were dropped"
                    );
                },
            }
        }
        None
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
