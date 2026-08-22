use crate::{DownloadError, traits::ActiveDownloadGeneration};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendEvent {
    Completed {
        generation: ActiveDownloadGeneration,
    },
    Error {
        generation: ActiveDownloadGeneration,
        error: DownloadError,
    },
}

impl BackendEvent {
    pub fn completed(generation: ActiveDownloadGeneration) -> Self {
        Self::Completed {
            generation,
        }
    }

    pub fn error(
        generation: ActiveDownloadGeneration,
        error: DownloadError,
    ) -> Self {
        Self::Error {
            generation,
            error,
        }
    }
}
