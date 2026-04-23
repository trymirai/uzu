#[bindings::export(Error)]
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError {
    #[error("Tokio error: {message}")]
    TokioError {
        message: String,
    },
    #[error(transparent)]
    Device(#[from] crate::device::DeviceError),
    #[error(transparent)]
    Storage(#[from] crate::storage::StorageError),
    #[error(transparent)]
    Registry(#[from] crate::registry::RegistryError),
    #[error("Unable to create backend")]
    UnableToCreateBackend {},
    #[error("Backend not found")]
    BackendNotFound {},
    #[error("Unable to get downloader progress stream")]
    UnableToGetDownloaderProgressStream {},
    #[error(transparent)]
    ChatSession(#[from] nagare::chat::ChatSessionError),
    #[error(transparent)]
    ClassificationSession(#[from] nagare::classification::ClassificationSessionError),
    #[error(transparent)]
    TextToSpeechSession(#[from] nagare::text_to_speech::TextToSpeechSessionError),
}
