use crate::{DataType, classifier::ClassifierError, config::ModelType, session::config::TtsRunConfigError};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TtsModelConfigError {
    #[error("expected model_type={expected:?}, got {actual:?}")]
    ModelTypeMismatch {
        expected: ModelType,
        actual: ModelType,
    },
    #[error("missing TTS model config")]
    MissingTtsModelConfig,
    #[error("FishAudio tensor {key} dtype mismatch: expected {expected:?}, got {actual:?}")]
    FishAudioTensorDataTypeMismatch {
        key: Box<str>,
        expected: DataType,
        actual: DataType,
    },
    #[error(
        "FishAudio decoder requires num_codebooks > 1 plus positive codebook_size and max_seq_len, got num_codebooks={num_codebooks}, codebook_size={codebook_size}, max_seq_len={max_seq_len}"
    )]
    FishAudioDecoderCoreParametersInvalid {
        num_codebooks: usize,
        codebook_size: usize,
        max_seq_len: usize,
    },
    #[error(
        "FishAudio decoder num_codebooks={decoder_num_codebooks} does not match audio num_codebooks={audio_num_codebooks}"
    )]
    FishAudioDecoderNumCodebooksMismatch {
        decoder_num_codebooks: usize,
        audio_num_codebooks: usize,
    },
    #[error(
        "FishAudio decoder codebook_size={codebook_size} must be at least audio residual codec cardinality={audio_codec_cardinality}"
    )]
    FishAudioDecoderCodebookSizeTooSmall {
        codebook_size: usize,
        audio_codec_cardinality: usize,
    },
    #[error("FishAudio semantic token range is invalid: begin={begin}, end={end}")]
    FishAudioSemanticTokenRangeInvalid {
        begin: i64,
        end: i64,
    },
    #[error("FishAudio semantic token range overflow: begin={begin}, end={end}")]
    FishAudioSemanticTokenRangeOverflow {
        begin: i64,
        end: i64,
    },
    #[error(
        "FishAudio semantic codec cardinality={semantic_cardinality} does not match audio semantic codec cardinality={audio_semantic_cardinality}"
    )]
    FishAudioSemanticCodecCardinalityMismatch {
        semantic_cardinality: usize,
        audio_semantic_cardinality: usize,
    },
    #[error("followup_count overflow in async chain")]
    AsyncChainFollowupCountOverflow,
    #[error("async chain token count {total_count} exceeds capacity {capacity}")]
    AsyncChainTokenCountExceedsCapacity {
        total_count: usize,
        capacity: usize,
    },
    #[error("vocab mask not supported for initial_token_count={initial_token_count} (expected 1 or 2)")]
    UnsupportedInitialTokenCountForVocabMask {
        initial_token_count: usize,
    },
    #[error("two-token vocab mask size overflow (row_words={row_words})")]
    TwoTokenVocabMaskSizeOverflow {
        row_words: usize,
    },
    #[error("token bitmask size overflow")]
    TokenBitmaskSizeOverflow,
    #[error(
        "token bitmask length {actual_words}, expected {expected_words} (token_count={token_count}, row_words={row_words})"
    )]
    TokenBitmaskLengthMismatch {
        actual_words: usize,
        expected_words: usize,
        token_count: usize,
        row_words: usize,
    },
    #[error("precomputed bitmask size overflow")]
    PrecomputedBitmaskSizeOverflow,
    #[error("precomputed bitmask length {actual_words}, expected {expected_words}")]
    PrecomputedBitmaskLengthMismatch {
        actual_words: usize,
        expected_words: usize,
    },
    #[error("vocab_limit resolved to 0")]
    VocabLimitResolvedToZero,
    #[error("inline bitmask size overflow (token_count={token_count}, row_words={row_words})")]
    InlineBitmaskSizeOverflow {
        token_count: usize,
        row_words: usize,
    },
    #[error("vocab_limit with token_count={token_count} requires a precomputed bitmask")]
    VocabLimitRequiresPrecomputedBitmask {
        token_count: usize,
    },
    #[error("model_dim {model_dim} exceeds u32 range")]
    ModelDimExceedsU32 {
        model_dim: usize,
    },
    #[error("hidden capture row offset overflow (token_count={token_count}, model_dim={model_dim})")]
    HiddenCaptureRowOffsetOverflow {
        token_count: usize,
        model_dim: usize,
    },
    #[error("hidden capture src offset overflow")]
    HiddenCaptureSourceOffsetOverflow,
    #[error(
        "hidden capture shape/dtype mismatch: expected {expected_shape:?} @ {expected_data_type:?}, got {actual_shape:?} @ {actual_data_type:?}"
    )]
    HiddenCaptureTensorMismatch {
        expected_shape: Box<[usize]>,
        expected_data_type: DataType,
        actual_shape: Box<[usize]>,
        actual_data_type: DataType,
    },
    #[error(
        "override embedding shape/dtype mismatch: expected {expected_shape:?} @ {expected_data_type:?}, got {actual_shape:?} @ {actual_data_type:?}"
    )]
    OverrideEmbeddingTensorMismatch {
        expected_shape: Box<[usize]>,
        expected_data_type: DataType,
        actual_shape: Box<[usize]>,
        actual_data_type: DataType,
    },
    #[error("add_scale total_len overflow (token_count={token_count}, model_dim={model_dim})")]
    AddScaleTotalLengthOverflow {
        token_count: usize,
        model_dim: usize,
    },
    #[error("add_scale total_len {total_len} exceeds u32 range")]
    AddScaleTotalLengthExceedsU32 {
        total_len: usize,
    },
    #[error(
        "add_scale bias shape/dtype mismatch: expected {expected_shape:?} @ {expected_data_type:?}, got {actual_shape:?} @ {actual_data_type:?}"
    )]
    AddScaleBiasTensorMismatch {
        expected_shape: Box<[usize]>,
        expected_data_type: DataType,
        actual_shape: Box<[usize]>,
        actual_data_type: DataType,
    },
    #[error("async chain copy src_offset overflow (src_slot={src_slot})")]
    AsyncChainCopySourceOffsetOverflow {
        src_slot: usize,
    },
    #[error("async chain copy size overflow (count={count})")]
    AsyncChainCopySizeOverflow {
        count: usize,
    },
    #[error("async chain result slot {slot} out of bounds (capacity={capacity})")]
    AsyncChainResultSlotOutOfBounds {
        slot: usize,
        capacity: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TtsPromptConfigError {
    #[error("FishAudio prompt template references message.{field}, but default_message_fields does not provide it")]
    MissingDefaultMessageField {
        field: Box<str>,
    },
}

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
    #[error("Invalid TTS model config: {0}")]
    InvalidTtsModelConfig(#[from] TtsModelConfigError),
    #[error("Invalid TTS run config: {0}")]
    InvalidTtsRunConfig(#[from] TtsRunConfigError),
    #[error("Invalid TTS prompt config: {0}")]
    InvalidTtsPromptConfig(#[from] TtsPromptConfigError),
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
    #[error("No grammar backend available")]
    GrammarNoBackendAvailable,
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
