use std::{
    iter::{once, repeat_n},
    mem::replace,
    rc::Rc,
};

use crate::{
    backends::common::{
        Allocation, AllocationPool, AllocationType, Backend, Context, Encoder, Pending,
        gpu_types::trie::TrieNode as GpuTrieNode,
        kernel::{ContextRingUpdateKernel, TokenCopySampledKernel},
    },
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, sampling::SamplingMethod},
    engine::language_model::{
        LanguageModel,
        grammar::Grammar,
        state::LanguageModelState,
        stream::{LanguageModelStreamError, LanguageModelStreamOptions},
    },
    trie::TrieNode,
};

enum ForwardPassChaining<B: Backend> {
    Constant(u64),
    InFlight(DecodingStatePending<B>),
}

impl<B: Backend> ForwardPassChaining<B> {
    fn resolve<'grammar>(
        &mut self,
        tokens: &mut Vec<u64>,
        grammar: Option<&mut (dyn Grammar + 'grammar)>,
    ) -> Result<u64, LanguageModelStreamError<B>> {
        match self {
            Self::Constant(token_id) => Ok(*token_id),
            Self::InFlight(in_flight) => {
                assert!(in_flight.full_accept);
                for pending in replace(&mut in_flight.pending, Box::new([])) {
                    pending.wait_until_completed().map_err(LanguageModelStreamError::Backend)?;
                }
                let output = in_flight.output.as_slice::<u32>();
                assert_eq!(output.len(), 1);
                let token_id = output[0] as u64;
                *self = Self::Constant(token_id);
                tokens.push(token_id);
                if let Some(grammar) = grammar {
                    grammar.accept_token(token_id)?;
                }
                Ok(token_id)
            },
        }
    }
}

struct DecodingStatePending<B: Backend> {
    input_trie: TrieNode,
    full_accept: bool,
    pending: Box<[Pending<B>]>,
    output: Allocation<B>,
}

enum DecodingState<B: Backend> {
    Seeded {
        seed_token: u64,
    },
    ForwardPassPending(DecodingStatePending<B>),
    Accepting {
        full: Box<[(usize, u64)]>,
        accepted: usize,
    },
    Invalid,
}

pub struct LanguageModelStream<'a, B: Backend> {
    model: &'a LanguageModel<B>,
    model_state: &'a mut LanguageModelState<B>,
    options: LanguageModelStreamOptions<'a>,
    allocation_pool: Rc<AllocationPool<B>>,
    context_ring: Option<Allocation<B>>,
    decoding_state: DecodingState<B>,
}

impl<'a, B: Backend> LanguageModelStream<'a, B> {
    pub fn new(
        model: &'a LanguageModel<B>,
        input: &[u64],
        model_state: &'a mut LanguageModelState<B>,
        mut options: LanguageModelStreamOptions<'a>,
    ) -> Result<Self, LanguageModelStreamError<B>> {
        if model_state.tokens.is_empty() && input.is_empty() {
            return Err(LanguageModelStreamError::NoSeedToken);
        };

        if options.speculator.is_some() && !model.speculation_supported() {
            return Err(LanguageModelStreamError::SpeculatorsNotSupported);
        }

        let allocation_pool = Rc::new(model.context.create_allocation_pool(false));

        let mut context_ring =
            if let Some(suffix_repetition_length) = options.sampling_method.suffix_repetition_length() {
                let mut context_ring = model
                    .context
                    .create_allocation(
                        (2 + suffix_repetition_length) * DataType::U32.size_in_bytes(),
                        AllocationType::Global,
                    )
                    .map_err(LanguageModelStreamError::Backend)?;

                let state_tokens_range =
                    model_state.tokens.len().saturating_sub(suffix_repetition_length)..model_state.tokens.len();

                context_ring.copyin(
                    &once(0) // offset
                        .chain(once(state_tokens_range.len() as u64)) // length
                        .chain(model_state.tokens[state_tokens_range.clone()].iter().copied()) // tokens
                        .chain(repeat_n(0, suffix_repetition_length - state_tokens_range.len())) // pad if not full
                        .map(|x| x as u32)
                        .collect::<Box<[_]>>(),
                );

                Some(context_ring)
            } else {
                None
            };

        let decoding_state = if !input.is_empty() {
            model_state.last_output_token.take();

            let mut encoder = Encoder::<B>::new_with_pool(&model.context, allocation_pool.clone())
                .map_err(LanguageModelStreamError::Backend)?;

            let input_trie = TrieNode::flat(model_state.tokens.len(), input, &model_state.prng);
            let input_flat_trie = input_trie.linearize();

            let mut token_ids = encoder
                .allocate_constant(input.len() * DataType::U64.size_in_bytes())
                .map_err(LanguageModelStreamError::Backend)?;
            token_ids.copyin(&input);

            let input_flat_trie_nodes = input_flat_trie.token_subtrie_ranges().collect::<Box<[GpuTrieNode]>>();
            let full_accept = true;
            let batch_dim = BatchTopology::new(&input_flat_trie_nodes, full_accept);

            model_state
                .transformer_state
                .prepare(model_state.transformer_state.context_length(), batch_dim.size(), &model.context)
                .map_err(LanguageModelStreamError::Backend)?;

            let logits = model
                .decoder
                .encode(
                    &token_ids,
                    &batch_dim,
                    Some((input.len() - 1)..input.len()),
                    &mut model_state.transformer_state,
                    &mut encoder,
                )?
                .unwrap();

            let seeds = if matches!(options.sampling_method, SamplingMethod::Stochastic { .. }) {
                let mut seeds = encoder
                    .allocate_constant(DataType::U64.size_in_bytes())
                    .map_err(LanguageModelStreamError::Backend)?;
                seeds.copyin(&[model_state.prng.derive((model_state.tokens.len() + input.len() - 1) as u64)]);
                Some(seeds)
            } else {
                None
            };

            let bitmask = if let Some(grammar) = options.grammar.as_deref_mut() {
                let mut bitmask = encoder
                    .allocate_constant(
                        model.vocab_size.div_ceil(DataType::U32.size_in_bits()) * DataType::U32.size_in_bytes(),
                    )
                    .map_err(LanguageModelStreamError::Backend)?;

                if grammar.next_bitmask(bitmask.as_slice_mut()) {
                    Some(bitmask)
                } else {
                    None
                }
            } else {
                None
            };

            let output = model
                .sampling
                .encode(
                    &logits,
                    seeds.as_ref(),
                    bitmask.as_ref(),
                    context_ring.as_ref(),
                    Some(&token_ids),
                    &options.sampling_method,
                    &batch_dim,
                    (input.len() - 1)..input.len(),
                    &mut encoder,
                )
                .map_err(LanguageModelStreamError::Backend)?;

            drop(bitmask);
            drop(seeds);
            drop(logits);

            model_state
                .transformer_state
                .encode_accept(&(0..input.len()).collect::<Box<[usize]>>(), &mut encoder)
                .map_err(LanguageModelStreamError::Backend)?;

            if let Some(suffix_repetition_length) = options.sampling_method.suffix_repetition_length() {
                model.context_ring_update.encode(
                    &token_ids,
                    context_ring.as_mut().unwrap(),
                    suffix_repetition_length as u32,
                    input.len() as u32,
                    &mut encoder,
                );
            }

            drop(token_ids);

            model_state.tokens.extend(input);

            let pending = Box::new([encoder.end_encoding().submit()]);

            DecodingState::ForwardPassPending(DecodingStatePending {
                input_trie,
                full_accept,
                pending,
                output,
            })
        } else {
            // TODO: this leaks previous LanguageModelStreamOptions
            DecodingState::Seeded {
                seed_token: model_state.last_output_token.take().unwrap(),
            }
        };

        Ok(LanguageModelStream {
            model,
            model_state,
            options,
            allocation_pool,
            context_ring,
            decoding_state,
        })
    }

    fn generate(&mut self) -> Result<Option<u64>, LanguageModelStreamError<B>> {
        let (mut prev_output, encoder): (ForwardPassChaining<B>, Option<Encoder<B>>) =
            match replace(&mut self.decoding_state, DecodingState::Invalid) {
                DecodingState::Seeded {
                    seed_token,
                } => (ForwardPassChaining::Constant(seed_token), None),
                DecodingState::ForwardPassPending(forward_pass_pending) => {
                    if forward_pass_pending.full_accept {
                        (ForwardPassChaining::InFlight(forward_pass_pending), None)
                    } else {
                        for pending in forward_pass_pending.pending {
                            pending.wait_until_completed().map_err(LanguageModelStreamError::Backend)?;
                        }
                        let sampled_tokens = forward_pass_pending
                            .output
                            .as_slice::<u32>()
                            .iter()
                            .map(|x| *x as u64)
                            .collect::<Box<[u64]>>();
                        let full = forward_pass_pending
                            .input_trie
                            .linearize()
                            .accept(&sampled_tokens, self.options.grammar.as_deref_mut())?;
                        self.decoding_state = DecodingState::Accepting {
                            full,
                            accepted: 0,
                        };
                        return self.generate();
                    }
                },
                DecodingState::Accepting {
                    full,
                    accepted,
                } => {
                    let token_id = full[accepted].1;

                    if accepted < full.len() - 1 {
                        self.decoding_state = DecodingState::Accepting {
                            full,
                            accepted: accepted + 1,
                        };
                        return Ok(Some(token_id));
                    } else {
                        let token_indicies = full.iter().map(|(i, _)| *i).collect::<Box<[usize]>>();
                        let token_ids = full.iter().map(|(_, t)| *t).collect::<Box<[u64]>>();
                        let mut encoder =
                            Encoder::<B>::new_with_pool(&self.model.context, self.allocation_pool.clone())
                                .map_err(LanguageModelStreamError::Backend)?;
                        self.model_state
                            .transformer_state
                            .encode_accept(&token_indicies, &mut encoder)
                            .map_err(LanguageModelStreamError::Backend)?;
                        if let Some(suffix_repetition_length) = self.options.sampling_method.suffix_repetition_length()
                        {
                            let mut token_ids_const = encoder
                                .allocate_constant(full.len() * DataType::U64.size_in_bytes())
                                .map_err(LanguageModelStreamError::Backend)?;
                            token_ids_const.copyin(&token_ids);
                            self.model.context_ring_update.encode(
                                &token_ids_const,
                                self.context_ring.as_mut().unwrap(),
                                suffix_repetition_length as u32,
                                full.len() as u32,
                                &mut encoder,
                            );
                        }
                        self.model_state.tokens.extend(&token_ids[..token_ids.len() - 1]);
                        (ForwardPassChaining::Constant(token_id), Some(encoder))
                    }
                },
                DecodingState::Invalid => unreachable!(),
            };

        let context_length = self.model_state.transformer_state.context_length();

        let mut pending = Vec::new();
        let mut encoder = if let Some(encoder) = encoder {
            encoder
        } else {
            Encoder::<B>::new_with_pool(&self.model.context, self.allocation_pool.clone())
                .map_err(LanguageModelStreamError::Backend)?
        };

        let (input_trie, chain_copy, full_accept) = if let Some(speculator) = &self.options.speculator {
            let seed = prev_output.resolve(&mut self.model_state.tokens, self.options.grammar.as_deref_mut())?;

            let prefix = self.model_state.tokens().iter().copied().chain(once(seed)).collect::<Box<[u64]>>();

            let input_trie = TrieNode::from_speculator(
                &prefix,
                &self.model_state.prng,
                self.options.grammar.as_deref_mut(),
                speculator.speculator,
                self.model.vocab_size,
                &speculator.trie_creation_config,
                speculator.speculation_budget,
            );

            (input_trie, None, false)
        } else {
            let (token, chain_copy) = match &prev_output {
                ForwardPassChaining::Constant(token) => (*token, None),
                ForwardPassChaining::InFlight(pending) => (0, Some(&pending.output)),
            };
            (TrieNode::new(token, self.model_state.prng.derive(context_length as u64)), chain_copy, true)
        };
        let input_flat_trie = input_trie.linearize();

        let token_ids = if let Some(chain_copy) = chain_copy {
            let mut token_ids =
                encoder.allocate_scratch(DataType::U64.size_in_bytes()).map_err(LanguageModelStreamError::Backend)?;
            self.model.token_copy.encode(chain_copy, &mut token_ids, &mut encoder);
            token_ids
        } else {
            let mut token_ids = encoder
                .allocate_constant(input_flat_trie.len() * DataType::U64.size_in_bytes())
                .map_err(LanguageModelStreamError::Backend)?;
            token_ids.copyin(&input_flat_trie.token_ids().collect::<Box<[u64]>>());
            token_ids
        };

        let input_flat_trie_nodes = input_flat_trie.token_subtrie_ranges().collect::<Box<[GpuTrieNode]>>();
        let batch_dim = BatchTopology::new(&input_flat_trie_nodes, full_accept);

        self.model_state
            .transformer_state
            .prepare(self.model_state.transformer_state.context_length(), batch_dim.size(), &self.model.context)
            .map_err(LanguageModelStreamError::Backend)?;

        let logits = self
            .model
            .decoder
            .encode(
                &token_ids,
                &batch_dim,
                Some(0..batch_dim.size()),
                &mut self.model_state.transformer_state,
                &mut encoder,
            )?
            .unwrap();

        let (bitmask, mut encoder) = if let Some(grammar) = self.options.grammar.as_deref_mut() {
            if chain_copy.is_some() {
                pending.push(encoder.end_encoding().submit());

                let mut encoder = Encoder::<B>::new_with_pool(&self.model.context, self.allocation_pool.clone())
                    .map_err(LanguageModelStreamError::Backend)?;

                let mut bitmask = encoder
                    .allocate_constant(
                        self.model.vocab_size.div_ceil(DataType::U32.size_in_bits()) * DataType::U32.size_in_bytes(),
                    )
                    .map_err(LanguageModelStreamError::Backend)?;

                prev_output.resolve(&mut self.model_state.tokens, Some(grammar))?;
                if grammar.next_bitmask(bitmask.as_slice_mut()) {
                    (Some(bitmask), encoder)
                } else {
                    (None, encoder)
                }
            } else {
                let mut bitmasks = encoder
                    .allocate_constant(
                        input_flat_trie.len()
                            * self.model.vocab_size.div_ceil(DataType::U32.size_in_bits())
                            * DataType::U32.size_in_bytes(),
                    )
                    .map_err(LanguageModelStreamError::Backend)?;

                if input_flat_trie.fill_bitmasks(bitmasks.as_slice_mut(), self.model.vocab_size, grammar) {
                    (Some(bitmasks), encoder)
                } else {
                    (None, encoder)
                }
            }
        } else {
            (None, encoder)
        };

        let seeds = if matches!(self.options.sampling_method, SamplingMethod::Stochastic { .. }) {
            let mut seeds = encoder
                .allocate_constant(input_flat_trie.len() * DataType::U64.size_in_bytes())
                .map_err(LanguageModelStreamError::Backend)?;
            seeds.copyin(&input_flat_trie.token_seeds().collect::<Box<[u64]>>());
            Some(seeds)
        } else {
            None
        };

        let output = self
            .model
            .sampling
            .encode(
                &logits,
                seeds.as_ref(),
                bitmask.as_ref(),
                self.context_ring.as_ref(),
                Some(&token_ids),
                &self.options.sampling_method,
                &batch_dim,
                0..batch_dim.size(),
                &mut encoder,
            )
            .map_err(LanguageModelStreamError::Backend)?;

        drop(bitmask);
        drop(seeds);
        drop(logits);

        if full_accept {
            self.model_state
                .transformer_state
                .encode_accept(&(0..batch_dim.size()).collect::<Box<[usize]>>(), &mut encoder)
                .map_err(LanguageModelStreamError::Backend)?;

            if let Some(suffix_repetition_length) = self.options.sampling_method.suffix_repetition_length() {
                self.model.context_ring_update.encode(
                    &token_ids,
                    self.context_ring.as_mut().unwrap(),
                    suffix_repetition_length as u32,
                    batch_dim.size() as u32,
                    &mut encoder,
                );
            }
        }

        drop(token_ids);

        pending.push(encoder.end_encoding().submit());

        self.decoding_state = DecodingState::ForwardPassPending(DecodingStatePending {
            input_trie,
            full_accept,
            pending: pending.into_boxed_slice(),
            output,
        });

        Ok(Some(prev_output.resolve(&mut self.model_state.tokens, self.options.grammar.as_deref_mut())?))
    }
}

impl<'a, B: Backend> Iterator for LanguageModelStream<'a, B> {
    type Item = Result<u64, LanguageModelStreamError<B>>;

    fn next(&mut self) -> Option<Result<u64, LanguageModelStreamError<B>>> {
        self.generate().transpose()
    }
}

impl<'a, B: Backend> Drop for LanguageModelStream<'a, B> {
    fn drop(&mut self) {
        // todo!()
        // TODO:
    }
}

unsafe impl<'a, B: Backend> Send for LanguageModelStream<'a, B> {}
