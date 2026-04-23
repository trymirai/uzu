#[bindings::export(Error)]
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum KeyringError {
    #[error("Backend error: {message}")]
    BackendError {
        message: String,
    },
}
