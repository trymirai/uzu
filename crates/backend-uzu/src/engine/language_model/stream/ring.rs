use std::iter::{once, repeat_n};

use crate::{
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder, Pending, kernel::TokenCopySampledKernel,
    },
    data_type::DataType,
    encodable_block::{mixer::MixerTokenTopology, sampling::SamplingMethod},
    engine::language_model::{
        LanguageModel,
        state::LanguageModelState,
        stream::{LanguageModelStreamError, LanguageModelStreamOptions},
    },
};

// Very bad code, but fast and relatively complete (only grammar + speculators missing + context ring off-by-one remaining)

pub struct ForwardPassPending<B: Backend> {
    command_buffer: Pending<B>,
    output: Allocation<B>,
}

pub struct RingDecodingLoop<'a, B: Backend> {
    language_model: &'a LanguageModel<B>,
    state: &'a mut LanguageModelState<B>,
    options: LanguageModelStreamOptions<'a>,
    context_ring: Option<Allocation<B>>,
    forward_pass_pending: Option<ForwardPassPending<B>>,
}

impl<'a, B: Backend> RingDecodingLoop<'a, B> {
    pub fn new(
        language_model: &'a LanguageModel<B>,
        input: &[u64],
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions<'a>,
    ) -> Result<Self, LanguageModelStreamError<B>> {
        let context_ring = if let Some(suffix_repetition_length) = options.sampling_method.suffix_repetition_length() {
            let mut context_ring = language_model
                .context
                .create_allocation(
                    (suffix_repetition_length + 2) * DataType::U32.size_in_bytes(),
                    AllocationType::Global,
                )
                .map_err(LanguageModelStreamError::Backend)?;

            let input_tokens_range = input.len().saturating_sub(suffix_repetition_length)..input.len();
            let state_tokens_range =
                state.tokens.len().saturating_sub(suffix_repetition_length.saturating_sub(input.len()))
                    ..state.tokens.len();

            context_ring.copyin(
                &once(0) // offset
                    .chain(once((state_tokens_range.len() + input_tokens_range.len()) as u64)) // length
                    .chain(state.tokens[state_tokens_range.clone()].iter().copied()) // state tokens
                    .chain(input[input_tokens_range.clone()].iter().copied()) // input tokens
                    .chain(repeat_n(
                        0,
                        suffix_repetition_length - (state_tokens_range.len() + input_tokens_range.len()),
                    )) // pad if not full
                    .map(|x| x as u32)
                    .collect::<Box<[_]>>(),
            );

            Some(context_ring)
        } else {
            None
        };

        let forward_pass_pending = if !input.is_empty() {
            state.next_seed_token = None;

            state
                .transformer_state
                .prepare(state.tokens.len(), input.len(), &language_model.context)
                .map_err(LanguageModelStreamError::Backend)?;

            let mut encoder = Encoder::<B>::new(&language_model.context).map_err(LanguageModelStreamError::Backend)?;

            let mut token_ids = encoder
                .allocate_constant(input.len() * DataType::U64.size_in_bytes())
                .map_err(LanguageModelStreamError::Backend)?;
            token_ids.copyin(&input);
            let mut sampling_token_ids =
                encoder.allocate_constant(DataType::U64.size_in_bytes()).map_err(LanguageModelStreamError::Backend)?;
            sampling_token_ids.copyin(&[*input.last().unwrap()]);

            let seeds = if matches!(options.sampling_method, SamplingMethod::Stochastic { .. }) {
                let mut seeds = encoder
                    .allocate_constant(DataType::U64.size_in_bytes())
                    .map_err(LanguageModelStreamError::Backend)?;
                seeds.copyin(&[state.prng.derive((state.tokens.len() + input.len() - 1) as u64)]);
                Some(seeds)
            } else {
                None
            };

            let logits = language_model
                .decoder
                .encode(
                    &token_ids,
                    input.len(),
                    Some(input.len() - 1..input.len()),
                    &MixerTokenTopology::Flat,
                    &mut state.transformer_state,
                    &mut encoder,
                )?
                .unwrap();
            let output = language_model
                .sampling
                .encode(
                    &logits,
                    seeds.as_ref(),
                    None,
                    context_ring.as_ref(),
                    Some(&sampling_token_ids),
                    &options.sampling_method,
                    1,
                    &mut encoder,
                )
                .map_err(LanguageModelStreamError::Backend)?;

            drop(logits);
            drop(token_ids);

            state
                .transformer_state
                .encode_accept(&(0..input.len()).collect::<Box<[_]>>(), &mut encoder)
                .map_err(LanguageModelStreamError::Backend)?;

            state.tokens.extend(input);

            Some(ForwardPassPending {
                command_buffer: encoder.end_encoding().submit(),
                output,
            })
        } else if !state.tokens.is_empty() {
            None
        } else {
            return Err(LanguageModelStreamError::NoSeedToken);
        };

        Ok(Self {
            language_model,
            state,
            options,
            context_ring,
            forward_pass_pending,
        })
    }

    fn generate(&mut self) -> Result<Option<u64>, LanguageModelStreamError<B>> {
        self.state
            .transformer_state
            .prepare(self.state.tokens.len(), 1, &self.language_model.context)
            .map_err(LanguageModelStreamError::Backend)?;

        let mut encoder = Encoder::<B>::new(&self.language_model.context).map_err(LanguageModelStreamError::Backend)?;

        let (prev_token_ids, cur_token_ids_alloc, cur_token_ids_rust, command_buffer) = if let Some(pending) =
            self.forward_pass_pending.take()
        {
            let mut token_ids =
                encoder.allocate_constant(DataType::U64.size_in_bytes()).map_err(LanguageModelStreamError::Backend)?;
            let token_copy = if self.context_ring.is_some() {
                &self.language_model.token_copy_ring
            } else {
                &self.language_model.token_copy_plain
            };
            token_copy.encode(
                &pending.output,
                &mut token_ids,
                self.context_ring.as_mut(),
                self.options.sampling_method.suffix_repetition_length().map(|x| x as u32),
                &mut encoder,
            );
            (Some(pending.output), token_ids, None, Some(pending.command_buffer))
        } else {
            let token_id_rust = self.state.next_seed_token.take().unwrap();
            let mut token_ids =
                encoder.allocate_constant(DataType::U64.size_in_bytes()).map_err(LanguageModelStreamError::Backend)?;
            token_ids.copyin(&[token_id_rust]);
            (None, token_ids, Some(token_id_rust), None)
        };

        let seeds = if matches!(self.options.sampling_method, SamplingMethod::Stochastic { .. }) {
            let mut seeds =
                encoder.allocate_constant(DataType::U64.size_in_bytes()).map_err(LanguageModelStreamError::Backend)?;
            seeds.copyin(&[self.state.prng.derive(self.state.tokens.len() as u64)]);
            Some(seeds)
        } else {
            None
        };

        let logits = self
            .language_model
            .decoder
            .encode(
                &cur_token_ids_alloc,
                1,
                Some(0..1),
                &MixerTokenTopology::Flat,
                &mut self.state.transformer_state,
                &mut encoder,
            )?
            .unwrap();

        let output = self
            .language_model
            .sampling
            .encode(
                &logits,
                seeds.as_ref(),
                None,
                self.context_ring.as_ref(),
                Some(&cur_token_ids_alloc),
                &self.options.sampling_method,
                1,
                &mut encoder,
            )
            .map_err(LanguageModelStreamError::Backend)?;
        drop(logits);
        drop(cur_token_ids_alloc);

        self.state.transformer_state.encode_accept(&[0], &mut encoder).map_err(LanguageModelStreamError::Backend)?;

        self.forward_pass_pending = Some(ForwardPassPending {
            command_buffer: encoder.end_encoding().submit(),
            output,
        });

        let this_token = if let Some(command_buffer) = command_buffer {
            command_buffer.wait_until_completed().map_err(LanguageModelStreamError::Backend)?;
            prev_token_ids.unwrap().copyout::<u32>()[0] as u64
        } else {
            cur_token_ids_rust.unwrap()
        };

        self.state.tokens.push(this_token);

        Ok(Some(this_token))
    }
}

impl<'a, B: Backend> Iterator for RingDecodingLoop<'a, B> {
    type Item = Result<u64, LanguageModelStreamError<B>>;

    fn next(&mut self) -> Option<Result<u64, LanguageModelStreamError<B>>> {
        self.generate().transpose()
    }
}

impl<'a, B: Backend> Drop for RingDecodingLoop<'a, B> {
    fn drop(&mut self) {
        if let Some(pending) = self.forward_pass_pending.take() {
            pending.command_buffer.wait_until_completed().unwrap();
            self.state.next_seed_token = Some(pending.output.copyout::<u32>()[0] as u64);
        }
    }
}
