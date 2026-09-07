use shoji::types::model::ModelIdentifier;
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

use crate::{
    engine::EngineError,
    helpers::SharedAccess,
    storage::{DownloadPhase, DownloadState, Storage, StorageError},
};

#[bindings::export(Class)]
#[derive(Clone)]
pub struct Downloader {
    identifier: ModelIdentifier,
    storage: SharedAccess<Storage>,
}

impl Downloader {
    pub fn new(
        identifier: ModelIdentifier,
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
            _ => Ok(self.storage.lock().await.download(&self.identifier).await?),
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
        if !state.is_in_progress() {
            return Ok(DownloaderStream::empty(self.identifier.clone()));
        }
        Ok(DownloaderStream::new(self.identifier.clone(), self.storage.lock().await.subscribe()))
    }
}

#[bindings::export(Class(Stream))]
#[derive(Clone)]
pub struct DownloaderStream {
    identifier: ModelIdentifier,
    stream: SharedAccess<Option<BroadcastStream<(ModelIdentifier, DownloadState)>>>,
}

impl DownloaderStream {
    pub fn new(
        identifier: ModelIdentifier,
        stream: BroadcastStream<(ModelIdentifier, DownloadState)>,
    ) -> Self {
        Self {
            identifier,
            stream: SharedAccess::new(Some(stream)),
        }
    }

    pub fn empty(identifier: ModelIdentifier) -> Self {
        Self {
            identifier,
            stream: SharedAccess::new(None),
        }
    }
}

#[bindings::export(Implementation)]
impl DownloaderStream {
    #[bindings::export(Method(StreamNext))]
    pub async fn next(&self) -> Option<DownloadState> {
        let mut stream_guard = self.stream.lock().await;
        let stream = stream_guard.as_mut()?;
        while let Some(result) = stream.next().await {
            match result {
                Ok((identifier, state)) => {
                    if identifier == self.identifier {
                        if !state.is_in_progress() {
                            *stream_guard = None;
                        }
                        return Some(state);
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
