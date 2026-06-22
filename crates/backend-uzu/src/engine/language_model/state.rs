use thiserror::Error;

use crate::{
    backends::common::Backend,
    encodable_block::transformer::TransformerState,
    engine::language_model::{LanguageModel, prng::PRng},
};

pub struct LanguageModelState<B: Backend> {
    pub(super) tokens: Vec<u64>,
    pub(super) next_seed_token: Option<u64>,
    pub(super) prng: PRng,
    pub(super) transformer_state: TransformerState<B>,
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
        let next_seed_token = None;

        let prng = PRng::new(rand::random());

        let transformer_state = self
            .decoder
            .create_empty_state(max_context_length, &self.context)
            .map_err(LanguageModelCreateEmptyStateError::Backend)?;

        Ok(LanguageModelState {
            tokens,
            next_seed_token,
            prng,
            transformer_state,
        })
    }
}
