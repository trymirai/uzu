use std::{pin::Pin, sync::Arc};

use tokenizers::Tokenizer;

use crate::{
    traits::backend::{Error, Instance as InstanceTrait},
    types::{
        basic::SamplingParameters,
        session::chat::{ChatConfig, ChatReplyConfig},
    },
};

#[derive(Debug)]
pub struct TokenScores {
    pub log_probabilities: Box<[f64]>,
    pub label_mass: f64,
    pub prefill_duration: std::time::Duration,
}

#[derive(Debug, thiserror::Error)]
pub enum TokenScoreError {
    #[error("numeric scoring is not supported by this backend")]
    Unsupported,
    #[error("prompt exceeds the context limit")]
    ContextOverflow,
    #[error("invalid scoring request: {0}")]
    InvalidInput(String),
    #[error("invalid model output: {0}")]
    InvalidModelOutput(String),
    #[error("scoring backend failed: {0}")]
    Backend(String),
}

pub enum TokenStreamOutput {
    LimitReached, // This should be just end of stream
    Token(u64),
}

#[derive(Debug, Clone, Default)]
pub struct TokenStreamMetrics {
    pub num_prefill_forward_passes: usize,
    pub num_decode_forward_passes: usize,
    pub num_tokens_prefilled: usize,
    pub num_tokens_proposed: usize,
    pub num_tokens_accepted: usize,
    pub num_tokens_returned: usize,
}

pub type StreamInput = Vec<u64>;
pub type StreamOutput = TokenStreamOutput;
pub type StreamMetrics = Option<TokenStreamMetrics>;

pub trait Backend: Send + Sync {
    fn instance<'a>(
        &'a self,
        reference: String,
        config: ChatConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn Instance>, Error>> + Send + 'a>>;
}

pub trait Instance:
    InstanceTrait<
        StreamConfig = ChatReplyConfig,
        StreamInput = StreamInput,
        StreamOutput = StreamOutput,
        StreamMetrics = StreamMetrics,
    >
{
    fn with_config(
        &self,
        config: ChatConfig,
    ) -> Box<dyn Instance>;

    fn tokenizer(&self) -> Arc<Tokenizer>;

    fn max_context_length(&self) -> Option<usize>;

    fn stop_token_ids(&self) -> Option<Box<[u64]>>;

    fn sampling_defaults(&self) -> SamplingParameters;

    fn score_token_ids<'a>(
        &'a self,
        _prompt: &'a [u64],
        _labels: &'a [u64],
    ) -> Pin<Box<dyn Future<Output = Result<TokenScores, TokenScoreError>> + Send + 'a>> {
        Box::pin(async { Err(TokenScoreError::Unsupported) })
    }
}
