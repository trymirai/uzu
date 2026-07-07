use std::{
    iter::{once, repeat_n},
    mem::replace,
    rc::Rc,
};

use shoji::traits::backend::chat_token::TokenStreamMetrics;

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
        full: Box<[(usize, u64, u64)]>,
        num_accepted: usize,
    },
    Halted,
    Invalid,
}

pub struct LanguageModelStream<'a, B: Backend> {
    model: &'a LanguageModel<B>,
    model_state: &'a mut LanguageModelState<B>,
    options: LanguageModelStreamOptions<'a>,
    allocation_pool: Rc<AllocationPool<B>>,
    context_ring: Option<Allocation<B>>,
    decoding_state: DecodingState<B>,
    metrics: TokenStreamMetrics,
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

        if model_state
            .max_context_length
            .is_some_and(|max_context_length| model_state.tokens.len() + input.len() > max_context_length)
        {
            return Err(LanguageModelStreamError::ContextOverflow);
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

        let mut metrics = TokenStreamMetrics::default();

        let decoding_state = if !input.is_empty() {
            model_state.last_output_token.take();

            // NOTE: this is required for attention correctness (hardcoded suffix 1024). This is really bad design, attention should be rewritten to allow on-demand suffix length
            let max_batch_size = 1024;
            let number_of_batches = input.len().div_ceil(max_batch_size);

            model_state
                .transformer_state
                .prepare(
                    model_state.transformer_state.context_length() + (number_of_batches - 1) * max_batch_size,
                    usize::min(max_batch_size, input.len()),
                    &model.context,
                )
                .map_err(LanguageModelStreamError::Backend)?;

            let mut encoder = Encoder::<B>::new_with_pool(&model.context, allocation_pool.clone())
                .map_err(LanguageModelStreamError::Backend)?;

            let mut output = None;

            for (batch_idx, input_chunk) in input.chunks(max_batch_size).enumerate() {
                let last_batch = batch_idx == number_of_batches - 1;

                let input_trie = TrieNode::flat(model_state.tokens.len(), input_chunk, &model_state.prng);
                let input_flat_trie = input_trie.linearize();

                let mut token_ids = encoder
                    .allocate_constant(input_chunk.len() * DataType::U64.size_in_bytes())
                    .map_err(LanguageModelStreamError::Backend)?;
                token_ids.copyin(input_chunk);

                let input_flat_trie_nodes = input_flat_trie.token_subtrie_ranges().collect::<Box<[GpuTrieNode]>>();
                let batch_dim = BatchTopology::new(&input_flat_trie_nodes, true);

                let logits = model
                    .decoder
                    .encode(
                        &token_ids,
                        &batch_dim,
                        last_batch.then(|| (input_chunk.len() - 1)..input_chunk.len()),
                        &mut model_state.transformer_state,
                        &mut encoder,
                        &[],
                    )?
                    .logits;

                if last_batch {
                    let logits = logits.unwrap();

                    let seeds = if matches!(options.sampling_method, SamplingMethod::Stochastic { .. }) {
                        let mut seeds = encoder
                            .allocate_constant(DataType::U64.size_in_bytes())
                            .map_err(LanguageModelStreamError::Backend)?;
                        seeds.copyin(&[model_state
                            .prng
                            .derive((model_state.tokens.len() + input_chunk.len() - 1) as u64)]);
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

                    output = Some(
                        model
                            .sampling
                            .encode(
                                &logits,
                                seeds.as_ref(),
                                bitmask.as_ref(),
                                context_ring.as_ref(),
                                Some(&token_ids),
                                &options.sampling_method,
                                &batch_dim,
                                (batch_dim.size() - 1)..batch_dim.size(),
                                &mut encoder,
                            )
                            .map_err(LanguageModelStreamError::Backend)?,
                    );
                }

                model_state
                    .transformer_state
                    .encode_accept(&(0..input_chunk.len()).collect::<Box<[usize]>>(), &mut encoder)
                    .map_err(LanguageModelStreamError::Backend)?;

                if let Some(suffix_repetition_length) = options.sampling_method.suffix_repetition_length() {
                    model.context_ring_update.encode(
                        &token_ids,
                        context_ring.as_mut().unwrap(),
                        suffix_repetition_length as u32,
                        input_chunk.len() as u32,
                        &mut encoder,
                    );
                }

                model_state.tokens.extend(input_chunk);
            }

            let pending = Box::new([encoder.end_encoding().submit()]);

            metrics.num_forward_passes += 1;
            metrics.num_tokens_prefilled += input.len();
            metrics.num_tokens_proposed += 1;
            metrics.num_tokens_accepted += 1;

            DecodingState::ForwardPassPending(DecodingStatePending {
                input_trie: TrieNode::new(0, 0),
                full_accept: true,
                pending,
                output: output.unwrap(),
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
            metrics,
        })
    }

    fn generate(&mut self) -> Result<Option<u64>, LanguageModelStreamError<B>> {
        let (mut prev_output, encoder): (ForwardPassChaining<B>, Option<Encoder<B>>) =
            match replace(&mut self.decoding_state, DecodingState::Invalid) {
                DecodingState::Seeded {
                    seed_token,
                } => {
                    self.model_state.tokens.push(seed_token);
                    if let Some(grammar) = self.options.grammar.as_deref_mut() {
                        let _ = grammar.accept_token(seed_token); // TODO: this should not be ignored
                    }
                    self.metrics.num_tokens_returned += 1;
                    (ForwardPassChaining::Constant(seed_token), None)
                },
                DecodingState::ForwardPassPending(forward_pass_pending) => {
                    if forward_pass_pending.full_accept {
                        self.metrics.num_tokens_returned += 1;
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
                        self.metrics.num_tokens_accepted += full.len();
                        self.decoding_state = DecodingState::Accepting {
                            full,
                            num_accepted: 0,
                        };
                        return self.generate();
                    }
                },
                DecodingState::Accepting {
                    full,
                    num_accepted,
                } => {
                    let output_token_id = full[num_accepted].2;

                    self.metrics.num_tokens_returned += 1;

                    if num_accepted < full.len() - 1 {
                        self.decoding_state = DecodingState::Accepting {
                            full,
                            num_accepted: num_accepted + 1,
                        };
                        return Ok(Some(output_token_id));
                    } else {
                        let accepted_token_indicies = full.iter().map(|(i, _, _)| *i).collect::<Box<[usize]>>();
                        let accepted_input_token_ids = full.iter().map(|(_, t, _)| *t).collect::<Box<[u64]>>();
                        let accepted_output_token_ids = full.iter().map(|(_, _, t)| *t).collect::<Box<[u64]>>();
                        let mut encoder =
                            Encoder::<B>::new_with_pool(&self.model.context, self.allocation_pool.clone())
                                .map_err(LanguageModelStreamError::Backend)?;
                        self.model_state
                            .transformer_state
                            .encode_accept(&accepted_token_indicies, &mut encoder)
                            .map_err(LanguageModelStreamError::Backend)?;
                        if let Some(suffix_repetition_length) = self.options.sampling_method.suffix_repetition_length()
                        {
                            let mut accepted_input_token_ids_const = encoder
                                .allocate_constant(full.len() * DataType::U64.size_in_bytes())
                                .map_err(LanguageModelStreamError::Backend)?;
                            accepted_input_token_ids_const.copyin(&accepted_input_token_ids);
                            self.model.context_ring_update.encode(
                                &accepted_input_token_ids_const,
                                self.context_ring.as_mut().unwrap(),
                                suffix_repetition_length as u32,
                                full.len() as u32,
                                &mut encoder,
                            );
                        }
                        self.model_state.tokens.extend(accepted_output_token_ids);
                        (ForwardPassChaining::Constant(output_token_id), Some(encoder))
                    }
                },
                DecodingState::Halted => return Ok(None),
                DecodingState::Invalid => unreachable!(),
            };

        let context_length = self.model_state.transformer_state.context_length();

        if self.model_state.max_context_length.is_some_and(|max_context_length| context_length >= max_context_length) {
            self.decoding_state = DecodingState::Halted;
            return Ok(Some(prev_output.resolve(&mut self.model_state.tokens, self.options.grammar.as_deref_mut())?));
        }

        let mut pending = Vec::new();
        let mut encoder = if let Some(encoder) = encoder {
            encoder
        } else {
            Encoder::<B>::new_with_pool(&self.model.context, self.allocation_pool.clone())
                .map_err(LanguageModelStreamError::Backend)?
        };

        let (input_trie, chain_copy, full_accept) = if let Some(speculator) = &self.options.speculator {
            prev_output.resolve(&mut self.model_state.tokens, self.options.grammar.as_deref_mut())?;

            let input_trie = TrieNode::from_speculator(
                &self.model_state.tokens,
                &self.model_state.prng,
                self.options.grammar.as_deref_mut(),
                speculator.speculator,
                self.model.vocab_size,
                &speculator.trie_creation_config,
                speculator.speculation_budget,
                self.model_state.max_context_length.map(|max_context_length| max_context_length - context_length),
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
                &[],
            )?
            .logits
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

        drop(seeds);
        drop(bitmask);
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

        self.metrics.num_forward_passes += 1;
        self.metrics.num_tokens_proposed += input_flat_trie.len();
        if full_accept {
            self.metrics.num_tokens_accepted += input_flat_trie.len();
        }

        self.decoding_state = DecodingState::ForwardPassPending(DecodingStatePending {
            input_trie,
            full_accept,
            pending: pending.into_boxed_slice(),
            output,
        });

        Ok(Some(prev_output.resolve(&mut self.model_state.tokens, self.options.grammar.as_deref_mut())?))
    }

    pub fn metrics(&self) -> &TokenStreamMetrics {
        &self.metrics
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
        let last_output_token = match replace(&mut self.decoding_state, DecodingState::Invalid) {
            DecodingState::Seeded {
                seed_token,
            } => Some(seed_token),
            DecodingState::ForwardPassPending(in_flight) => {
                for pending in in_flight.pending {
                    pending.wait_until_completed().unwrap();
                }

                if !in_flight.full_accept {
                    let mut encoder =
                        Encoder::<B>::new_with_pool(&self.model.context, self.allocation_pool.clone()).unwrap();
                    self.model_state.transformer_state.encode_accept(&[0], &mut encoder).unwrap();
                    encoder.end_encoding().submit().wait_until_completed().unwrap();
                }

                Some(in_flight.output.as_slice::<u32>()[0] as u64)
            },
            DecodingState::Accepting {
                full,
                num_accepted,
            } => {
                assert!(num_accepted > 0 && num_accepted < full.len());

                let mut encoder =
                    Encoder::<B>::new_with_pool(&self.model.context, self.allocation_pool.clone()).unwrap();
                self.model_state
                    .transformer_state
                    .encode_accept(
                        &full.iter().take(num_accepted + 1).map(|(i, _, _)| *i).collect::<Box<[usize]>>(),
                        &mut encoder,
                    )
                    .unwrap();
                encoder.end_encoding().submit().wait_until_completed().unwrap();

                self.model_state.tokens.extend(full.iter().take(num_accepted).map(|(_, _, t)| *t));

                Some(full[num_accepted].2)
            },
            DecodingState::Halted => None,
            DecodingState::Invalid => None, // TODO: proper error handling
        };

        self.model_state.last_output_token = last_output_token;
    }
}

unsafe impl<'a, B: Backend> Send for LanguageModelStream<'a, B> {} // TODO: this should be done properly
