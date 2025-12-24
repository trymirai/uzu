use crate::backends::metal::error::ClassifierError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Model folder not found")]
    ModelFolderNotFound,
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
    #[error("Unsupported context mode for model")]
    UnsupportedContextModeForModel,
    #[error("Unsupported speculator config for model")]
    UnsupportedSpeculatorConfigForModel,
    #[error("Language model generator not loaded")]
    LanguageModelGeneratorNotLoaded,
    #[error("Unable to load prompt template")]
    UnableToLoadPromptTemplate,
    #[error("Unable to render prompt template")]
    UnableToRenderPromptTemplate,
    #[error("Only assistant messages can have reasoning content")]
    UnexpectedReasoningContent,
    #[error("Unable to build output parser regex")]
    UnableToBuildOutputParserRegex,
    #[error("Unable to encode text")]
    UnableToEncodeText,
    #[error("Unable to decode text")]
    UnableToDecodeText,
    #[error("Context length exceeded")]
    ContextLengthExceeded,
    #[error("Prefill failed")]
    PrefillFailed,
    #[error("Generate failed")]
    GenerateFailed,
    #[error("Sampling failed")]
    SamplingFailed,
    #[error("Grammar error")]
    GrammarError,
    #[error("Grammar rejected the token")]
    GrammarReject,
    #[error("Token {0} out of grammar vocabulary range (0..{1})")]
    TokenOutOfGrammarRange(u64, usize),
    #[error("Classifier error: {0}")]
    Classifier(#[from] ClassifierError),
}
