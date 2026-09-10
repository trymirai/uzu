use crate::backends::DownloadGeneration;

#[derive(Debug)]
pub enum BackendEvent {
    Completed {
        generation: DownloadGeneration,
    },
    Error {
        generation: DownloadGeneration,
        message: String,
    },
}

impl BackendEvent {
    pub fn generation(&self) -> DownloadGeneration {
        match self {
            Self::Completed {
                generation,
            }
            | Self::Error {
                generation,
                ..
            } => *generation,
        }
    }
}
