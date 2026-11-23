use std::{pin::Pin, sync::Arc};

use tokenizers::Tokenizer;

use crate::{
    traits::backend::{Error, Instance as InstanceTrait},
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

pub struct StreamConfig {
    pub reply: ChatReplyConfig,
    /// Generated marker after which grammar constraints apply; absent means immediate enforcement.
    pub grammar_trigger: Option<String>,
}

pub struct StreamImage {
    pub url: String,
    /// Encoded image file contents read from `url`
    pub bytes: Vec<u8>,
}

#[derive(Default)]
pub struct StreamInput {
    pub tokens: Vec<u64>,
    pub images: Vec<StreamImage>,
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
        StreamConfig = StreamConfig,
        StreamInput = StreamInput,
        StreamOutput = StreamOutput,
        StreamMetrics = StreamMetrics,
    >
{
    fn with_config(
        &self,
        config: ChatConfig,
    ) -> Box<dyn Instance>;

    fn with_image_pad_token_id(
        &self,
        image_pad_token_id: u64,
    ) -> Result<Box<dyn Instance>, Error> {
        Err(format!("Backend does not support image pad token {image_pad_token_id}").into())
    }

    fn tokenizer(&self) -> Arc<Tokenizer>;

    fn max_context_length(&self) -> Option<usize>;

    fn stop_token_ids(&self) -> Option<Box<[u64]>>;

    fn sampling_defaults(&self) -> SamplingParameters;
}
