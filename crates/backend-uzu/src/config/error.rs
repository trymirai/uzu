use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Missing required field: {0}")]
    MissingField(String),
    #[error("No layers in transformer config")]
    NoLayers,
    #[error("Invalid config: {0}")]
    Invalid(String),
}
