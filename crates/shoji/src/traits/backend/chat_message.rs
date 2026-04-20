use std::pin::Pin;

use serde::{Deserialize, Serialize};

use crate::{
    traits::backend::{Error, Instance as InstanceTrait, chat::StreamConfig},
    types::{basic::Value, encoding::Message},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Output {
    Content(String),
    Reasoning(String),
    ToolCalls(Vec<Value>),
}

pub type Config = ();
pub type StreamInput = Vec<Message>;
pub type StreamOutput = Output;

pub trait Backend: Send + Sync {
    fn instance(
        &self,
        reference: String,
        config: Config,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn Instance>, Error>> + Send + '_>>;
}

pub trait Instance:
    InstanceTrait<StreamConfig = StreamConfig, StreamInput = StreamInput, StreamOutput = StreamOutput>
{
}

impl<T> Instance for T where
    T: InstanceTrait<StreamConfig = StreamConfig, StreamInput = StreamInput, StreamOutput = StreamOutput>
{
}
