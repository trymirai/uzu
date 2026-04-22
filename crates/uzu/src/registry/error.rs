#[bindings::export(Error)]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RegistryError {
    #[error("Unable to create: {message}")]
    UnableToCreate {
        message: String,
    },
    #[error("Unable to get models: {message}")]
    UnableToGetModels {
        message: String,
    },
    #[error("Unable to add registry: {identifier}")]
    UnableToAddRegistry {
        identifier: String,
    },
}
