#[bindings::export(Error)]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Unsupported device")]
    UnsupportedDevice,
}
