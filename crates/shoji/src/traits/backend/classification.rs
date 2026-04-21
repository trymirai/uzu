use std::pin::Pin;

use crate::{
    traits::backend::{Error, Instance as InstanceTrait},
    types::session::classification::{Message, Output},
};

pub type Config = ();
pub type StreamConfig = ();
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
