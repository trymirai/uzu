use std::{pin::Pin, sync::Arc};

use tokenizers::Tokenizer;

use crate::{
    traits::{
        State,
        backend::{Error, Instance as InstanceTrait},
    },
    types::{
        basic::SamplingParameters,
        session::chat::{ChatConfig, ChatReplyConfig},
    },
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
}

pub struct StreamInput {
    pub tokens: Vec<u64>,
    /// Context position (counting tokens already in the state) at which to snapshot the state while prefilling
    /// `tokens`, so that `Instance::rewind` can return there later.
    pub snapshot_position: Option<usize>,
}

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

    fn sampling_defaults(&self) -> SamplingParameters;

    /// Prepares `state` to process `tokens`, a whole new context: keeps the longest prefix the state can reuse and
    /// returns its length (the caller streams only the rest). Returns None when the state must be reset.
    fn rewind(
        &self,
        state: &mut dyn State,
        tokens: &[u64],
    ) -> Result<Option<usize>, Error>;
}
