#[bindings::export(Error)]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum DeviceError {
    #[error("Unsupported device")]
    UnsupportedDevice {},
}
