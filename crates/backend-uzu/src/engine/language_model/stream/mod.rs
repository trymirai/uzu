mod ring;
pub mod sync;

use thiserror::Error;

use crate::{
    backends::common::Backend,
    encodable_block::decoder::DecoderError,
    engine::language_model::{
        LanguageModel, grammar::CompiledGrammar, state::LanguageModelState, stream::ring::RingDecodingLoop,
    },
    speculators::speculator::Speculator,
};
pub use crate::{encodable_block::sampling::SamplingMethod, trie::TrieCreationConfig};

pub struct LanguageModelStreamSpeculatorOptions<'a> {
    pub speculator: &'a dyn Speculator,
    pub speculation_budget: usize,
    pub trie_creation_config: TrieCreationConfig,
}

pub struct LanguageModelStreamOptions<'a> {
    pub sampling_method: SamplingMethod,
    pub grammar: Option<&'a mut dyn CompiledGrammar>,
    pub speculator: Option<LanguageModelStreamSpeculatorOptions<'a>>,
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
    pub fn default_stream_options<'a>(&'a self) -> LanguageModelStreamOptions<'a> {
        LanguageModelStreamOptions {
            sampling_method: self.default_sampling_method(),
            grammar: None,
            speculator: None,
        }
    }

    pub fn stream<'a>(
        &'a self,
        input: &[u64],
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions<'a>,
    ) -> Result<impl Iterator<Item = Result<u64, LanguageModelStreamError<B>>> + 'a, LanguageModelStreamError<B>> {
        RingDecodingLoop::new(self, input, state, options)
    }
}
