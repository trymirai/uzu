#[bindings::export(Error)]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DeviceError {
    #[error("Unsupported device")]
    UnsupportedDevice,
}
