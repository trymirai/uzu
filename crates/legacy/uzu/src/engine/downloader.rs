use std::sync::Arc;

use shoji::types::model::ModelIdentifier;

use crate::{
    engine::{DownloaderStream, EngineError},
    storage::{DownloadState, Storage},
};

#[bindings::export(Class)]
#[derive(Clone)]
pub struct Downloader {
    identifier: ModelIdentifier,
    storage: Arc<Storage>,
}

impl Downloader {
    pub fn new(
        identifier: ModelIdentifier,
        storage: Arc<Storage>,
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
        self.storage.state(&self.identifier).await.ok()
    }

    #[bindings::export(Method)]
    pub async fn resume(&self) -> Result<(), EngineError> {
        Ok(self.storage.download(&self.identifier).await?)
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
        let events = self.storage.subscribe();
        let state = self.storage.state(&self.identifier).await?;
        if !state.is_in_progress() {
            return Ok(DownloaderStream::empty(self.identifier.clone()));
        }
        Ok(DownloaderStream::new(self.identifier.clone(), events))
    }
}
