use thiserror::Error;

use crate::{
    backends::common::Backend,
    encodable_block::{decoder::DecoderError, sampling::SamplingMethod},
    engine::language_model::{LanguageModel, state::LanguageModelState},
};

mod sync;

pub use sync::LanguageModelStreamDriver;

#[derive(Debug, Clone)]
pub struct LanguageModelStreamOptions {
    pub sampling_method: SamplingMethod,
    pub stop_token_ids: Vec<u64>,
    pub token_limit: Option<usize>,
}

impl LanguageModelStreamOptions {
    pub fn with_token_limit(
        mut self,
        token_limit: Option<usize>,
    ) -> Self {
        self.token_limit = token_limit;
        self
    }
}

#[derive(Debug, Error)]
pub enum LanguageModelStreamError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
    #[error("No seed token (both state and input are empty)")]
    NoSeedToken,
}

impl<B: Backend> LanguageModel<B> {
    pub fn default_stream_options(&self) -> LanguageModelStreamOptions {
        LanguageModelStreamOptions {
            sampling_method: self.default_sampling_method(),
            stop_token_ids: self.default_stop_token_ids().to_vec(),
            token_limit: None,
        }
    }

    pub fn stream<'a>(
        &'a self,
        input: &[u64],
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<impl Iterator<Item = Result<u64, LanguageModelStreamError<B>>> + 'a, LanguageModelStreamError<B>> {
        sync::LanguageModelIterator::new(self, input, state, options)
    }
}
