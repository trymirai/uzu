use std::iter::once;

use crate::{
    backends::common::{Backend, Encoder},
    encodable_block::{mixer::MixerTokenTopology, sampling::SamplingMethod},
    engine::language_model::{LanguageModel, state::LanguageModelState, stream::LanguageModelStreamError},
};

// Slow, awful, horrendous decoding loop just to test the new refactor

pub(crate) struct LanguageModelIteratorData {
    tokens: Vec<u64>,
    generated: usize,
    stopped: bool,
}

impl LanguageModelIteratorData {
    pub fn new<B: Backend>(
        input: &[u64],
        state: &LanguageModelState<B>,
    ) -> Result<Self, LanguageModelStreamError<B>> {
        if state.tokens.is_empty() && input.is_empty() {
            return Err(LanguageModelStreamError::NoSeedToken);
        }

        Ok(Self {
            tokens: input.to_vec(),
            generated: 0,
            stopped: false,
        })
    }

    pub fn step<B: Backend>(
        &mut self,
        model: &LanguageModel<B>,
        state: &mut LanguageModelState<B>,
    ) -> Result<Option<u64>, LanguageModelStreamError<B>> {
        if self.stopped {
            return Ok(None);
        }

        let seed_token = state.tokens.last().copied();
        let context_length = state.tokens.len() - usize::from(seed_token.is_some());
        let forward_pass_tokens = seed_token.clone().into_iter().chain(self.tokens.drain(..)).collect::<Box<[u64]>>();
        state
            .transformer_state
            .prepare(context_length, forward_pass_tokens.len(), &model.context)
            .map_err(LanguageModelStreamError::Backend)?;

        let mut encoder = Encoder::<B>::new(&model.context).map_err(LanguageModelStreamError::Backend)?;
        let mut token_ids = encoder
            .allocate_constant(forward_pass_tokens.len() * size_of::<u64>())
            .map_err(LanguageModelStreamError::Backend)?;
        token_ids.copyin(&forward_pass_tokens);

        let logits = model
            .decoder
            .encode(
                &token_ids,
                forward_pass_tokens.len(),
                Some((forward_pass_tokens.len() - 1)..forward_pass_tokens.len()),
                &MixerTokenTopology::Flat,
                &mut state.transformer_state,
                &mut encoder,
            )?
            .unwrap();

        let sampled_tokens = model
            .sampling
            .encode(&logits, None, None, None, None, &SamplingMethod::Greedy, 1, &mut encoder)
            .map_err(LanguageModelStreamError::Backend)?;

        state
            .transformer_state
            .encode_accept(&(0..forward_pass_tokens.len()).collect::<Vec<_>>(), &mut encoder)
            .map_err(LanguageModelStreamError::Backend)?;

        drop(logits);
        drop(token_ids);

        encoder.end_encoding().submit().wait_until_completed().map_err(LanguageModelStreamError::Backend)?;

        let sampled_token = sampled_tokens.copyout::<u32>()[0] as u64;

        state.tokens.extend(
            forward_pass_tokens
                .into_iter()
                .skip(if seed_token.is_some() {
                    1
                } else {
                    0
                })
                .chain(once(sampled_token)),
        );

        self.generated += 1;

        if model.generation_config.stop_token_ids.contains(&sampled_token)
            || state.transformer_state.max_context_length().is_some_and(|token_limit| self.generated >= token_limit)
        {
            self.stopped = true;
        }

        Ok(Some(sampled_token))
    }
}
