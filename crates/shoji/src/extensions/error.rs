#[bindings::export(Error)]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("Unable to save as wav: {message}")]
    UnableToSaveAsWav {
        message: String,
    },
}
