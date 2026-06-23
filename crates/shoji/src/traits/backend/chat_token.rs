use std::pin::Pin;

use crate::{
    traits::backend::{Error, Instance as InstanceTrait},
    types::session::chat::{ChatConfig, ChatReplyConfig, ChatReplyFinishReason},
};

pub type StreamInput = Vec<u64>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamOutput {
    Token(u64),
    Finished(ChatReplyFinishReason),
}

impl From<u64> for StreamOutput {
    fn from(token: u64) -> Self {
        Self::Token(token)
    }
}

pub trait Backend: Send + Sync {
    fn instance(
        &self,
        reference: String,
        config: ChatConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn Instance>, Error>> + Send + '_>>;
}

pub trait Instance:
    InstanceTrait<StreamConfig = ChatReplyConfig, StreamInput = StreamInput, StreamOutput = StreamOutput>
{
}

impl<T> Instance for T where
    T: InstanceTrait<StreamConfig = ChatReplyConfig, StreamInput = StreamInput, StreamOutput = StreamOutput>
{
}
