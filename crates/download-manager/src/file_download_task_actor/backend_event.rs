use crate::traits::ActiveDownloadGeneration;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BackendEvent {
    Completed {
        generation: ActiveDownloadGeneration,
    },
    Error {
        generation: ActiveDownloadGeneration,
        message: String,
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
        message: String,
    ) -> Self {
        Self::Error {
            generation,
            message,
        }
    }
}
