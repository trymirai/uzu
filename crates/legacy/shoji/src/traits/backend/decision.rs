use std::{pin::Pin, sync::Arc, time::Duration};

use tokenizers::Tokenizer;

use crate::{traits::backend::Error, types::session::classification::TokenCodecConfig};

pub struct DecisionProbabilities {
    pub probabilities: Vec<f32>,
    pub prefill_duration: Duration,
    pub memory_used_bytes: Option<usize>,
}

pub type Config = ();

pub trait DecisionBackend: Send + Sync {
    fn instance(
        &self,
        reference: String,
        config: Config,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn DecisionInstance>, Error>> + Send + '_>>;
}

pub trait DecisionInstance: Send + Sync {
    fn tokenizer(&self) -> Arc<Tokenizer>;

    fn token_codec_config(&self) -> TokenCodecConfig;

    // Answer codes in option order: option `i` of a question is shown under `codes()[i]`.
    fn codes(&self) -> Vec<String>;

    // Probabilities of the first `option_count` codes for one rendered question.
    fn probabilities<'a>(
        &'a self,
        input: &'a [u64],
        option_count: usize,
    ) -> Pin<Box<dyn Future<Output = Result<DecisionProbabilities, Error>> + Send + 'a>>;
}
