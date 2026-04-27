#[bindings::export(Error)]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum KeyringError {
    #[error("Backend error: {message}")]
    BackendError {
        message: String,
    },
    #[error(transparent)]
    Device(#[from] crate::device::DeviceError),
}
