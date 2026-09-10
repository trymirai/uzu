use shoji::types::model::ModelIdentifier;
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

use crate::{helpers::SharedAccess, storage::DownloadState};

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
