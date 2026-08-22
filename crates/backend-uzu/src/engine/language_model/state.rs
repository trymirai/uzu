use thiserror::Error;

#[cfg(grammar)]
use crate::engine::language_model::grammar::GrammarConfig;
use crate::{
    backends::common::Backend,
    encodable_block::{dflash::DFlashState, sampling::PRng, transformer::TransformerState},
    engine::language_model::LanguageModel,
};

pub(super) struct PendingOutput {
    pub(super) token: u64,
    #[cfg(grammar)]
    pub(super) grammar: Option<GrammarConfig>,
}

pub struct LanguageModelState<B: Backend> {
    pub(super) tokens: Vec<u64>,
    pub(super) last_output: Option<PendingOutput>,
    pub(super) prng: PRng,
    pub(super) transformer_state: TransformerState<B>,
    pub(super) speculator_state: Option<DFlashState<B>>,
    pub(super) max_context_length: Option<u32>,
    #[cfg(grammar)]
    pub(super) grammar_start: usize,
}

impl<B: Backend> LanguageModelState<B> {
    pub fn tokens(&self) -> &[u64] {
        &self.tokens
    }
}

#[derive(Debug, Error)]
pub enum LanguageModelCreateEmptyStateError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
}

impl<B: Backend> LanguageModel<B> {
    pub fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
    ) -> Result<LanguageModelState<B>, LanguageModelCreateEmptyStateError<B>> {
        let tokens = Vec::new();
        let last_output = None;

        let prng = PRng::new(rand::random());

        let transformer_state = self
            .decoder
            .create_empty_state(max_context_length, &self.engine.context)
            .map_err(LanguageModelCreateEmptyStateError::Backend)?;

        let speculator_state = self
            .speculator
            .as_ref()
            .map(|speculator| {
                speculator.empty_state(max_context_length.expect("speculator doesn't support unlimited state capacity"))
            })
            .transpose()
            .map_err(LanguageModelCreateEmptyStateError::Backend)?;

        Ok(LanguageModelState {
            #[cfg(grammar)]
            grammar_start: tokens.len(),
            tokens,
            last_output,
            prng,
            transformer_state,
            speculator_state,
            max_context_length,
        })
    }
}
