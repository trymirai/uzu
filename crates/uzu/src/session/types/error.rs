use crate::generator::error::GeneratorError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unable to create Metal context")]
    UnableToCreateMetalContext,
    #[error("Unable to load model configuration")]
    UnableToLoadConfig,
    #[error("Unable to load model weights")]
    UnableToLoadWeights,
    #[error("Unable to load tokenizer")]
    UnableToLoadTokenizer,
    #[error("Model is too large to fit into available RAM")]
    NotEnoughMemory,
    #[error("Generator not loaded. Call load() first")]
    GeneratorNotLoaded,
    #[error("Unable to load promot template")]
    UnableToLoadPromptTemplate,
    #[error("Unable to render prompt template")]
    UnableToRenderPromptTemplate,
    #[error("Unable to encode text")]
    UnableToEncodeText,
    #[error("Unable to decode text")]
    UnableToDecodeText,
}

impl From<GeneratorError> for Error {
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
