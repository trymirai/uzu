use thiserror::Error;

use crate::{
    backends::common::Backend,
    encodable_block::{sampling::PRng, transformer::TransformerState},
    engine::language_model::LanguageModel,
};

pub struct LanguageModelState<B: Backend> {
    pub(super) tokens: Vec<u64>,
    pub(super) last_output_token: Option<u64>, // TODO: this leaks previous LanguageModelStreamOptions
    pub(super) prng: PRng,
    pub(super) transformer_state: TransformerState<B>,
    pub(super) max_context_length: Option<usize>,
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
        max_context_length: Option<usize>,
    ) -> Result<LanguageModelState<B>, LanguageModelCreateEmptyStateError<B>> {
        let tokens = Vec::new();
        let last_output_token = None;

        let prng = PRng::new(rand::random());

        let transformer_state = self
            .decoder
            .create_empty_state(max_context_length, &self.context)
            .map_err(LanguageModelCreateEmptyStateError::Backend)?;

        Ok(LanguageModelState {
            tokens,
            last_output_token,
            prng,
            transformer_state,
            max_context_length,
        })
    }
}
