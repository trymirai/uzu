use std::pin::Pin;

use tokenizers::Tokenizer;

use crate::{
    traits::backend::{Error, Instance as InstanceTrait},
    types::session::chat::{ChatConfig, ChatReplyConfig},
};

pub type StreamInput = Vec<u64>;
pub type StreamOutput = u64;

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
}

impl<T> Instance for T where
    T: InstanceTrait<StreamConfig = ChatReplyConfig, StreamInput = StreamInput, StreamOutput = StreamOutput>
{
}
