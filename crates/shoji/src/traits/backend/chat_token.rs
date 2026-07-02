use std::pin::Pin;

use tokenizers::Tokenizer;

use crate::{
    traits::backend::{Error, Instance as InstanceTrait},
    types::session::chat::{ChatConfig, ChatReplyConfig},
};

pub enum TokenStreamOutput {
    PrefillStarted,
    PrefillFinished,
    Token(u64),
}

pub type StreamInput = Vec<u64>;
pub type StreamOutput = TokenStreamOutput;

pub trait Backend: Send + Sync {
    fn instance<'a>(
        &'a self,
        reference: String,
        config: ChatConfig,
        tokenizer: &'a Tokenizer,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn Instance>, Error>> + Send + 'a>>;
}

pub trait Instance:
    InstanceTrait<StreamConfig = ChatReplyConfig, StreamInput = StreamInput, StreamOutput = StreamOutput>
{
    fn max_context_length(&self) -> Option<usize>;

    fn stop_token_ids(&self) -> Option<Box<[u64]>>;
}
