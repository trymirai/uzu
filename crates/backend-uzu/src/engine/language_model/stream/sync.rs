use std::iter::once;

use crate::{
    backends::common::{Backend, Encoder},
    encodable_block::{mixer::MixerTokenTopology, sampling::SamplingMethod},
    engine::language_model::{
        LanguageModel,
        state::LanguageModelState,
        stream::{LanguageModelStreamError, LanguageModelStreamOptions},
    },
};

pub struct LanguageModelSyncStream<'a, B: Backend> {
    language_model: &'a LanguageModel<B>,
    state: &'a mut LanguageModelState<B>,
    options: LanguageModelStreamOptions,
    tokens: Vec<u64>,
    generated: usize,
    stopped: bool,
}

// Slow, awful, horrendous decoding loop just to test the new refactor

impl<'a, B: Backend> LanguageModelSyncStream<'a, B> {
    pub fn new(
        language_model: &'a LanguageModel<B>,
        input: &[u64],
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<Self, LanguageModelStreamError<B>> {
        if state.tokens.is_empty() && input.is_empty() {
            return Err(LanguageModelStreamError::NoSeedToken);
        }

        assert!(options.token_limit.is_none_or(|token_limit| token_limit > 0), "token limit 0 not supported yet");

        Ok(Self {
            language_model,
            state,
            options,
            tokens: input.to_vec(),
            generated: 0,
            stopped: false,
        })
    }

    fn generate(&mut self) -> Result<Option<u64>, LanguageModelStreamError<B>> {
        if self.stopped {
            return Ok(None);
        }

        let seed_token = self.state.tokens.last().copied();
        let context_length = self.state.tokens.len() - usize::from(seed_token.is_some());
        let forward_pass_tokens = seed_token.clone().into_iter().chain(self.tokens.drain(..)).collect::<Box<[u64]>>();

        assert!(!forward_pass_tokens.is_empty());

        self.state
            .transformer_state
            .prepare(context_length, forward_pass_tokens.len(), &self.language_model.context)
            .map_err(LanguageModelStreamError::Backend)?;

        let mut encoder = Encoder::<B>::new(&self.language_model.context).map_err(LanguageModelStreamError::Backend)?;

        let mut token_ids = encoder
            .allocate_constant(forward_pass_tokens.len() * size_of::<u64>())
            .map_err(LanguageModelStreamError::Backend)?;

        token_ids.copyin(&forward_pass_tokens);

        let logits = self
            .language_model
            .decoder
            .encode(
                &token_ids,
                forward_pass_tokens.len(),
                Some((forward_pass_tokens.len() - 1)..forward_pass_tokens.len()),
                &MixerTokenTopology::Flat,
                &mut self.state.transformer_state,
                &mut encoder,
            )?
            .unwrap();

        let sampled_tokens = self
            .language_model
            .sampling
            .encode(&logits, None, None, None, None, &SamplingMethod::Greedy, 1, &mut encoder)
            .map_err(LanguageModelStreamError::Backend)?;

        self.state
            .transformer_state
            .encode_accept(&(0..forward_pass_tokens.len()).collect::<Vec<_>>(), &mut encoder)
            .map_err(LanguageModelStreamError::Backend)?;

        drop(logits);
        drop(token_ids);

        encoder.end_encoding().submit().wait_until_completed().map_err(LanguageModelStreamError::Backend)?;

        let sampled_token = sampled_tokens.copyout::<u32>()[0] as u64;

        self.state.tokens.extend(
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

        if self.options.stop_token_ids.contains(&sampled_token)
            || self.options.token_limit.is_some_and(|token_limit| self.generated >= token_limit)
        {
            self.stopped = true;
        }

        Ok(Some(sampled_token))
    }
}

impl<'a, B: Backend> Iterator for LanguageModelSyncStream<'a, B> {
    type Item = Result<u64, LanguageModelStreamError<B>>;

    fn next(&mut self) -> Option<Result<u64, LanguageModelStreamError<B>>> {
        self.generate().transpose()
    }
}
