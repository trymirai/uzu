#[bindings::export(Error, name = "DeviceError")]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Unsupported device")]
    UnsupportedDevice,
}
