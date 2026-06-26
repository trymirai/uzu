use std::{collections::HashMap, pin::Pin};

use crate::traits::backend::{Error, Instance as InstanceTrait};

pub struct ClassifierOutput {
    pub logits: Vec<f32>,
    pub probabilities: HashMap<String, f32>,
}

pub type Config = ();
pub type StreamConfig = ();
pub type StreamInput = Vec<u64>;
pub type StreamOutput = ClassifierOutput;

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
