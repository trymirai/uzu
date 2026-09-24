use thiserror::Error;

use crate::{
    backends::common::{Backend, Encoder},
    encodable_block::{dflash::DFlashState, sampling::PRng, transformer::TransformerState},
    engine::language_model::LanguageModel,
};

pub struct LanguageModelState<B: Backend> {
    pub(super) tokens: Vec<u64>,
    pub(super) last_output_token: Option<u64>, // TODO: this leaks previous LanguageModelStreamOptions
    pub(super) prng: PRng,
    pub(super) transformer_state: TransformerState<B>,
    pub(super) speculator_state: Option<DFlashState<B>>,
    pub(super) max_context_length: Option<u32>,
    /// Number of leading `tokens` covered by the last snapshot of the transformer and speculator states, once the
    /// prefill that took it is submitted.
    pub(super) snapshot_position: Option<usize>,
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
        sampling_seed: u64,
    ) -> Result<LanguageModelState<B>, LanguageModelCreateEmptyStateError<B>> {
        let tokens = Vec::new();
        let last_output_token = None;

        let prng = PRng::new(sampling_seed);

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
            tokens,
            last_output_token,
            prng,
            transformer_state,
            speculator_state,
            max_context_length,
            snapshot_position: None,
        })
    }

    /// Prepares `state` to process `tokens`, a whole new context: keeps the longest prefix it can reuse, continuing
    /// where it is or returning to its snapshot, and returns that prefix's length, which is always shorter than
    /// `tokens`. Returns None when the state cannot be reused.
    pub fn rewind(
        &self,
        state: &mut LanguageModelState<B>,
        tokens: &[u64],
    ) -> Result<Option<usize>, B::Error> {
        // The last token is always prefilled again: the stream samples from it.
        let Some((_, context)) = tokens.split_last() else {
            return Ok(None);
        };
        let shared = state.tokens.iter().zip(context).take_while(|(a, b)| a == b).count();
        if shared == state.tokens.len() {
            return Ok(Some(shared));
        }
        let Some(position) = state.snapshot_position.filter(|&position| position <= shared) else {
            return Ok(None);
        };

        let mut encoder = Encoder::<B>::new(&self.engine.context)?;
        state.transformer_state.encode_restore(&mut encoder);
        assert_eq!(state.transformer_state.context_length() as usize, position);
        if let Some(speculator_state) = state.speculator_state.as_mut() {
            speculator_state.encode_restore(&mut encoder);
            assert_eq!(speculator_state.context_length() as usize, position);
        }
        encoder.end_encoding().submit().wait_until_completed()?;

        state.tokens.truncate(position);
        state.last_output_token = None;
        Ok(Some(position))
    }
}

#[cfg(test)]
#[path = "../../../unit/engine/language_model/rewind_test.rs"]
mod tests;
