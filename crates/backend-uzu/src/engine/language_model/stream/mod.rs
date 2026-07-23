use thiserror::Error;

use crate::{
    backends::common::Backend,
    encodable_block::decoder::DecoderError,
    engine::language_model::{
        LanguageModel,
        grammar::{Grammar, GrammarError},
        state::LanguageModelState,
    },
    speculators::dflash_speculator::DFlashTreeError,
};
pub use crate::{
    encodable_block::sampling::SamplingMethod, engine::language_model::stream::stream::LanguageModelStream,
};

mod stream;

pub struct LanguageModelStreamOptions {
    pub sampling_method: SamplingMethod,
    pub grammar: Option<Box<dyn Grammar>>,
}

#[derive(Debug, Error)]
pub enum LanguageModelStreamError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
    #[error("Grammar error: {0}")]
    Grammar(#[from] GrammarError),
    #[error("Speculator error: {0}")]
    Speculator(#[from] DFlashTreeError<B>),
    #[error("No seed token (both state and input are empty)")]
    NoSeedToken,
    #[error("Context overflow")]
    ContextOverflow,
}

impl<B: Backend> LanguageModel<B> {
    pub fn default_stream_options<'a>(&'a self) -> LanguageModelStreamOptions {
        LanguageModelStreamOptions {
            sampling_method: self.default_sampling_method(),
            grammar: None,
        }
    }

    pub fn stream<'a>(
        &'a self,
        input: &[u64],
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<LanguageModelStream<'a, B>, LanguageModelStreamError<B>> {
        LanguageModelStream::new(self, input, state, options)
    }
}
