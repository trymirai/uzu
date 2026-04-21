use std::pin::Pin;

use crate::{
    traits::backend::{Error, Instance as InstanceTrait},
    types::session::text_to_speech::PcmBatch,
};

pub type Config = ();
pub type StreamConfig = ();
pub type StreamInput = String;
pub type StreamOutput = PcmBatch;

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
