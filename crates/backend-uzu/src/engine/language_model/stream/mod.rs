use thiserror::Error;

#[cfg(grammar)]
use crate::engine::language_model::grammar::{Grammar, GrammarError};
use crate::{
    backends::common::Backend,
    encodable_block::decoder::DecoderError,
    engine::language_model::{LanguageModel, state::LanguageModelState},
    speculators::speculator::Speculator,
};
pub use crate::{
    encodable_block::sampling::SamplingMethod, engine::language_model::stream::stream::LanguageModelStream,
    trie::TrieCreationConfig,
};

mod stream;

pub struct LanguageModelStreamSpeculatorOptions<'a> {
    pub speculator: &'a dyn Speculator,
    pub speculation_budget: usize,
    pub trie_creation_config: TrieCreationConfig,
}

pub struct LanguageModelStreamOptions<'a> {
    pub sampling_method: SamplingMethod,
    #[cfg(grammar)]
    pub grammar: Option<Grammar>,
    pub speculator: Option<LanguageModelStreamSpeculatorOptions<'a>>,
}

#[derive(Debug, Error)]
pub enum LanguageModelStreamError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
    #[cfg(grammar)]
    #[error("Grammar error: {0}")]
    Grammar(#[from] GrammarError),
    #[error("Trie accept error: {0}")]
    TrieAccept(#[from] crate::trie::TrieAcceptError),
    #[error("Speculators are not supported by this model")]
    SpeculatorsNotSupported,
    #[error("No seed token (both state and input are empty)")]
    NoSeedToken,
    #[error("Context overflow")]
    ContextOverflow,
}

impl<B: Backend> LanguageModel<B> {
    pub fn default_stream_options<'a>(&'a self) -> LanguageModelStreamOptions<'a> {
        LanguageModelStreamOptions {
            sampling_method: self.default_sampling_method(),
            #[cfg(grammar)]
            grammar: None,
            speculator: None,
        }
    }

    pub fn stream<'a>(
        &'a self,
        input: &[u64],
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions<'a>,
    ) -> Result<LanguageModelStream<'a, B>, LanguageModelStreamError<B>> {
        LanguageModelStream::new(self, input, state, options)
    }
}
