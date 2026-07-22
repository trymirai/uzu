use std::pin::Pin;

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
    pub num_forward_passes: usize,
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
        tokenizer: Option<&'a Tokenizer>,
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
    fn max_context_length(&self) -> Option<usize>;

    fn stop_token_ids(&self) -> Option<Box<[u64]>>;
}
