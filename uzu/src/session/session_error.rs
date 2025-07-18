use crate::generator::error::GeneratorError;

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("Unable to create Metal context")]
    UnableToCreateMetalContext,
    #[error("Unable to load model configuration")]
    UnableToLoadConfig,
    #[error("Unable to load model weights")]
    UnableToLoadWeights,
    #[error("Unable to load tokenizer")]
    UnableToLoadTokenizer,
    #[error("Unable to load tokenizer configuration")]
    UnableToLoadTokenizerConfig,
    #[error("Model is too large to fit into available RAM")]
    UnsupportedModel,
}

impl From<GeneratorError> for SessionError {
    fn from(value: GeneratorError) -> Self {
        match value {
            GeneratorError::UnableToCreateMetalContext => {
                Self::UnableToCreateMetalContext
            },
            GeneratorError::UnableToLoadConfig => Self::UnableToLoadConfig,
            GeneratorError::UnableToLoadWeights => Self::UnableToLoadWeights,
        }
    }
}
