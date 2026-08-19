use std::{pin::Pin, sync::Arc};

use tokenizers::Tokenizer;

use crate::{
    traits::backend::{Error, Instance as InstanceTrait},
    types::session::chat::{ChatConfig, ChatReplyConfig},
};

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
    /// Wall time of the backend token-stream call that produced the first
    /// token (`next`), i.e. backend prefill as seen by the chat layer.
    pub prefill_duration: Option<std::time::Duration>,
    /// Cumulative wall time of backend token-stream calls after the first
    /// token, i.e. backend decode without parser/render overhead.
    pub decode_duration: Option<std::time::Duration>,
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
    fn tokenizer(&self) -> Arc<Tokenizer>;

    fn max_context_length(&self) -> Option<usize>;

    fn stop_token_ids(&self) -> Option<Box<[u64]>>;
}
