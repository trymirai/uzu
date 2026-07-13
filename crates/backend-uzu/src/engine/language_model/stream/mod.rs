use thiserror::Error;

use crate::{
    backends::common::Backend,
    encodable_block::{decoder::DecoderError, per_layer_embedding::PerLayerEmbeddingError},
    engine::language_model::{
        LanguageModel,
        grammar::{Grammar, GrammarError},
        state::LanguageModelState,
    },
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
    pub grammar: Option<Box<dyn Grammar>>,
    pub speculator: Option<LanguageModelStreamSpeculatorOptions<'a>>,
}

#[derive(Debug, Error)]
pub enum LanguageModelStreamError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
    #[error("Per-layer embedding error: {0}")]
    PerLayerEmbedding(#[from] PerLayerEmbeddingError<B>),
    #[error("Grammar error: {0}")]
    Grammar(#[from] GrammarError),
    #[error("Speculators are not supported by this model")]
    SpeculatorsNotSupported,
    #[error("No seed token (both state and input are empty)")]
    NoSeedToken,
    #[error("Context overflow")]
    ContextOverflow,
    #[error("Language-model state is unusable after a previous stream failure")]
    StatePoisoned,
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
    ) -> Result<LanguageModelStream<'a, B>, LanguageModelStreamError<B>> {
        LanguageModelStream::new(self, input, state, options)
    }
}
