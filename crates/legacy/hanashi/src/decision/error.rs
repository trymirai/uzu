#[derive(Debug, thiserror::Error)]
pub enum DecisionEncodingError {
    #[error("Duplicate option `{0}`")]
    DuplicateOption(String),
    #[error("{0}")]
    InvalidQuestion(String),
    #[error("Unable to encode text")]
    UnableToEncodeText,
    #[error("Unable to render prompt: {0}")]
    UnableToRenderPrompt(String),
    #[error("Decision prompts need a chat token codec")]
    UnsupportedTokenCodec,
}
