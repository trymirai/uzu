use std::pin::Pin;

use crate::{
    traits::backend::{Error, Instance as InstanceTrait},
    types::session::chat::{ChatConfig, ChatStreamConfig},
};

pub type StreamInput = Vec<u64>;
pub type StreamOutput = u64;

pub trait Backend: Send + Sync {
    fn instance(
        &self,
        reference: String,
        config: ChatConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn Instance>, Error>> + Send + '_>>;
}

pub trait Instance:
    InstanceTrait<StreamConfig = ChatStreamConfig, StreamInput = StreamInput, StreamOutput = StreamOutput>
{
}

impl<T> Instance for T where
    T: InstanceTrait<StreamConfig = ChatStreamConfig, StreamInput = StreamInput, StreamOutput = StreamOutput>
{
}
