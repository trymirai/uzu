#[bindings::export(Error)]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SettingsError {
    #[error("Backend error: {message}")]
    BackendError {
        message: String,
    },
}
