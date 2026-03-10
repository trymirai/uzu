use crate::classifier::ClassifierError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Model folder not found")]
    ModelFolderNotFound,
    #[error("Unable to open any backend")]
    UnableToOpenAnyBackend,
    #[error("Unable to create Context: {0}")]
    UnableToCreateContext(Box<dyn std::error::Error>),
    #[error("Unable to create CommandBuffer: {0}")]
    UnableToCreateCommandBuffer(Box<dyn std::error::Error>),
    #[error("Unable to load model configuration")]
    UnableToLoadConfig,
    #[error("Unable to load model weights")]
    UnableToLoadWeights,
    #[error("Unable to load tokenizer")]
    UnableToLoadTokenizer,
    #[error("Model is too large to fit into available RAM")]
    NotEnoughMemory,
    #[error("Command buffer failed: {0}")]
    CommandBufferFailed(Box<dyn std::error::Error>),
    #[error("Encode failed: {0}")]
    EncodeFailed(Box<dyn std::error::Error>),
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
    #[error("Grammar error: {0}")]
    GrammarError(String),
    #[error("Grammar rejected the token")]
    GrammarReject,
    #[error("Token {0} out of grammar vocabulary range (0..{1})")]
    TokenOutOfGrammarRange(u64, usize),
    #[error("Capture failed")]
    CaptureFailed,
    #[error("Classifier error: {0}")]
    Classifier(#[from] ClassifierError),
    #[error("Audio codec error: {0}")]
    AudioCodec(#[from] crate::audio::AudioError),
}
