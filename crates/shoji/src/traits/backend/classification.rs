use std::{collections::HashMap, convert::Infallible, pin::Pin, sync::Arc};

use tokenizers::Tokenizer;

use crate::{
    traits::backend::{Error, Instance as InstanceTrait},
    types::session::classification::TokenCodecConfig,
};

pub struct ClassifierOutput {
    pub logits: Vec<f32>,
    pub probabilities: HashMap<String, f32>,
}

pub type Config = ();
pub type StreamConfig = ();
pub type StreamInput = Vec<u64>;
pub type StreamOutput = ClassifierOutput;
pub type StreamMetrics = Option<Infallible>;

pub trait Backend: Send + Sync {
    fn instance(
        &self,
        reference: String,
        config: Config,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn Instance>, Error>> + Send + '_>>;
}

pub trait Instance:
    InstanceTrait<
        StreamConfig = StreamConfig,
        StreamInput = StreamInput,
        StreamOutput = StreamOutput,
        StreamMetrics = StreamMetrics,
    >
{
    fn tokenizer(&self) -> Arc<Tokenizer>;

    fn token_codec_config(&self) -> TokenCodecConfig;
}
