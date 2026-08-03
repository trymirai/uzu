use std::{
    iter::{once, repeat_n},
    mem::replace,
    sync::Arc,
};

use shoji::traits::backend::chat_token::TokenStreamMetrics;

#[cfg(grammar)]
use crate::engine::language_model::grammar::Grammar;
use crate::{
    backends::common::{
        Allocation, AllocationPool, AllocationType, Backend, Context, Encoder, Pending,
        gpu_types::trie::TrieNode as GpuTrieNode, kernel::ContextRingUpdateKernel,
    },
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, sampling::SamplingMethod},
    engine::{
        capture::CaptureSpan,
        language_model::{
            LanguageModel,
            state::LanguageModelState,
            stream::{LanguageModelStreamError, LanguageModelStreamOptions},
        },
    },
    speculators::dflash_speculator::DFlashTreeOptions,
    trie::TrieNode,
};

enum ForwardPassChaining<B: Backend> {
    Constant {
        token: u64,
        output_norm: Option<Allocation<B>>,
    },
    InFlight(DecodingStatePending<B>),
}

impl<B: Backend> ForwardPassChaining<B> {
    fn resolve<'a>(
        &'a mut self,
        tokens: &mut Vec<u64>,
        #[cfg(grammar)] grammar: Option<&mut Grammar>,
    ) -> Result<(u64, Option<&'a Allocation<B>>), LanguageModelStreamError<B>> {
        match self {
            Self::Constant {
                token,
                output_norm,
            } => Ok((*token, output_norm.as_ref())),
            Self::InFlight(in_flight) => {
                assert!(in_flight.full_accept);
                for pending in replace(&mut in_flight.pending, Box::new([])) {
                    pending.wait_until_completed().map_err(LanguageModelStreamError::Backend)?;
                }
                let output_tokens = in_flight.output_tokens.as_slice::<u32>();
                assert_eq!(output_tokens.len(), 1);
                let token_id = output_tokens[0] as u64;
                let output_norm = in_flight.output_norm.take();
                *self = Self::Constant {
                    token: token_id,
                    output_norm,
                };
                tokens.push(token_id);
                #[cfg(grammar)]
                if let Some(grammar) = grammar {
                    grammar.accept_token(token_id)?;
                }
                let Self::Constant {
                    output_norm,
                    ..
                } = self
                else {
                    unreachable!()
                };
                Ok((token_id, output_norm.as_ref()))
            },
        }
    }
}

struct DecodingStatePending<B: Backend> {
    input_trie: TrieNode,
    full_accept: bool,
    pending: Box<[Pending<B>]>,
    _capture_span: Option<CaptureSpan<B>>,
    hidden_features: Option<Box<[Allocation<B>]>>,
    output_norm: Option<Allocation<B>>,
    output_tokens: Allocation<B>,
}

enum DecodingState<B: Backend> {
    Seeded {
        seed_token: u64,
    },
    ForwardPassPending(DecodingStatePending<B>),
    Accepting {
        full: Box<[(usize, u64, u64)]>,
        num_accepted: usize,
        hidden_features: Option<Box<[Allocation<B>]>>,
        output_norm: Option<(Allocation<B>, usize)>,
    },
    Halted,
    Invalid,
}

fn prefill_chunk_parts(
    input_chunk: &[u64],
    last_batch: bool,
    split_logits_row: bool,
) -> [Option<(&[u64], bool)>; 2] {
    if last_batch && split_logits_row && input_chunk.len() > 1 {
        let (prompt_chunk, sample_chunk) = input_chunk.split_at(input_chunk.len() - 1);
        [Some((prompt_chunk, false)), Some((sample_chunk, true))]
    } else {
        [Some((input_chunk, last_batch)), None]
    }
}

pub struct LanguageModelStream<'a, B: Backend> {
    model: &'a LanguageModel<B>,
    model_state: &'a mut LanguageModelState<B>,
    options: LanguageModelStreamOptions,
    allocation_pool: Arc<AllocationPool<B>>,
    context_ring: Option<Allocation<B>>,
    decoding_state: DecodingState<B>,
    metrics: TokenStreamMetrics,
}

impl<'a, B: Backend> LanguageModelStream<'a, B> {
    pub fn new(
        model: &'a LanguageModel<B>,
        input: &[u64],
        model_state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<Self, LanguageModelStreamError<B>> {
        #[cfg(grammar)]
        let mut options = options;
        if model_state.tokens.is_empty() && input.is_empty() {
            return Err(LanguageModelStreamError::NoSeedToken);
        };

        if model_state
            .max_context_length
            .is_some_and(|max_context_length| model_state.tokens.len() + input.len() > max_context_length)
        {
            return Err(LanguageModelStreamError::ContextOverflow);
        }

        let capture_span = if let Some(capture_manager) = &model.engine.capture_manager
            && let Some(capture_request) = capture_manager.maybe_capture_prefill_step()
        {
            Some(capture_request.start().map_err(LanguageModelStreamError::Backend)?)
        } else {
            None
        };

        let allocation_pool = Arc::new(model.engine.context.create_allocation_pool(false));

        let mut context_ring =
            if let Some(suffix_repetition_length) = options.sampling_method.suffix_repetition_length() {
                let mut context_ring = model
                    .engine
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
                    &model.engine.context,
                )
                .map_err(LanguageModelStreamError::Backend)?;

            let mut encoder = Encoder::<B>::new_with_pool(&model.engine.context, allocation_pool.clone())
                .map_err(LanguageModelStreamError::Backend)?;

            let mut output_tokens = None;
            let mut output_norm = None;
            let split_logits_row = model.decoder.prefill_cache_skips_trailing_layers();
            let hidden_feature_layer_indices =
                model.speculator.as_ref().map(|speculator| speculator.hidden_feature_layer_indices());

            for (input_chunk, sample_last) in input
                .chunks(max_batch_size)
                .enumerate()
                .flat_map(|(batch_idx, input_chunk)| {
                    prefill_chunk_parts(input_chunk, batch_idx == number_of_batches - 1, split_logits_row)
                })
                .flatten()
            {
                let input_trie = TrieNode::flat(model_state.tokens.len(), input_chunk, &model_state.prng);
                let input_flat_trie = input_trie.linearize();

                let mut token_ids = encoder
                    .allocate_constant(input_chunk.len() * DataType::U32.size_in_bytes())
                    .map_err(LanguageModelStreamError::Backend)?;
                token_ids.copyin(&input_chunk.iter().map(|token_id| *token_id as u32).collect::<Box<[u32]>>());

                let input_flat_trie_nodes = input_flat_trie.token_subtrie_ranges().collect::<Box<[GpuTrieNode]>>();
                let batch_dim = BatchTopology::new(&input_flat_trie_nodes, true);

                let decoder_output = model.decoder.encode(
                    &token_ids,
                    &batch_dim,
                    sample_last.then(|| (input_chunk.len() - 1)..input_chunk.len()),
                    hidden_feature_layer_indices,
                    &mut model_state.transformer_state,
                    &mut encoder,
                )?;
                let logits = decoder_output.logits;

                if sample_last {
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

                    #[cfg(grammar)]
                    let bitmask = if let Some(grammar) = options.grammar.as_mut() {
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
                    #[cfg(not(grammar))]
                    let bitmask = None;

                    output_norm = decoder_output.final_hidden;
                    output_tokens = Some(
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

                if let Some(speculator) = model.speculator.as_ref() {
                    let speculator_state = model_state.speculator_state.as_mut().unwrap();
                    speculator
                        .append_state(
                            speculator_state,
                            decoder_output.hidden_features.as_ref().unwrap(),
                            &(0..input_chunk.len()).collect::<Box<[usize]>>(),
                            &mut encoder,
                        )
                        .map_err(LanguageModelStreamError::Backend)?;
                }

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
                _capture_span: capture_span,
                hidden_features: None,
                output_norm,
                output_tokens: output_tokens.unwrap(),
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
                    #[cfg(grammar)]
                    if let Some(grammar) = self.options.grammar.as_mut() {
                        let _ = grammar.accept_token(seed_token); // TODO: this should not be ignored
                    }
                    self.metrics.num_tokens_returned += 1;
                    (
                        ForwardPassChaining::Constant {
                            token: seed_token,
                            output_norm: None,
                        },
                        None,
                    )
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
                            .output_tokens
                            .as_slice::<u32>()
                            .iter()
                            .map(|x| *x as u64)
                            .collect::<Box<[u64]>>();
                        let flat_trie = forward_pass_pending.input_trie.linearize();
                        let full = flat_trie.accept(
                            &sampled_tokens,
                            #[cfg(grammar)]
                            self.options.grammar.as_mut(),
                        )?;
                        let output_norm = forward_pass_pending.output_norm.map(|norm| {
                            let row_bytes = norm.size() / flat_trie.len();
                            (norm, row_bytes)
                        });
                        self.metrics.num_tokens_accepted += full.len();
                        self.decoding_state = DecodingState::Accepting {
                            full,
                            num_accepted: 0,
                            hidden_features: forward_pass_pending.hidden_features,
                            output_norm,
                        };
                        return self.generate();
                    }
                },
                DecodingState::Accepting {
                    full,
                    num_accepted,
                    hidden_features,
                    output_norm,
                } => {
                    let output_token_id = full[num_accepted].2;

                    self.metrics.num_tokens_returned += 1;

                    if num_accepted < full.len() - 1 {
                        self.decoding_state = DecodingState::Accepting {
                            full,
                            num_accepted: num_accepted + 1,
                            hidden_features,
                            output_norm,
                        };
                        return Ok(Some(output_token_id));
                    } else {
                        let accepted_token_indicies = full.iter().map(|(i, _, _)| *i).collect::<Box<[usize]>>();
                        let accepted_input_token_ids = full.iter().map(|(_, t, _)| *t).collect::<Box<[u64]>>();
                        let accepted_output_token_ids = full.iter().map(|(_, _, t)| *t).collect::<Box<[u64]>>();
                        let mut encoder =
                            Encoder::<B>::new_with_pool(&self.model.engine.context, self.allocation_pool.clone())
                                .map_err(LanguageModelStreamError::Backend)?;
                        self.model_state
                            .transformer_state
                            .encode_accept(&accepted_token_indicies, &mut encoder)
                            .map_err(LanguageModelStreamError::Backend)?;
                        if let Some(speculator) = self.model.speculator.as_ref() {
                            speculator
                                .append_state(
                                    self.model_state.speculator_state.as_mut().unwrap(),
                                    hidden_features.as_deref().unwrap(),
                                    &accepted_token_indicies,
                                    &mut encoder,
                                )
                                .map_err(LanguageModelStreamError::Backend)?;
                        }
                        let output_norm = if let Some((final_hidden, row_bytes)) = output_norm {
                            let row = full.last().unwrap().0;
                            let mut norm =
                                encoder.allocate_scratch(row_bytes).map_err(LanguageModelStreamError::Backend)?;
                            encoder.encode_copy(&final_hidden, row * row_bytes..(row + 1) * row_bytes, &mut norm, ..);
                            Some(norm)
                        } else {
                            None
                        };
                        if let Some(suffix_repetition_length) = self.options.sampling_method.suffix_repetition_length()
                        {
                            let mut accepted_input_token_ids_const = encoder
                                .allocate_constant(full.len() * DataType::U32.size_in_bytes())
                                .map_err(LanguageModelStreamError::Backend)?;
                            accepted_input_token_ids_const.copyin(
                                &accepted_input_token_ids
                                    .iter()
                                    .map(|token_id| *token_id as u32)
                                    .collect::<Box<[u32]>>(),
                            );
                            self.model.context_ring_update.encode(
                                &accepted_input_token_ids_const,
                                self.context_ring.as_mut().unwrap(),
                                suffix_repetition_length as u32,
                                full.len() as u32,
                                &mut encoder,
                            );
                        }
                        self.model_state.tokens.extend(accepted_output_token_ids);
                        (
                            ForwardPassChaining::Constant {
                                token: output_token_id,
                                output_norm,
                            },
                            Some(encoder),
                        )
                    }
                },
                DecodingState::Halted => return Ok(None),
                DecodingState::Invalid => unreachable!(),
            };

        let context_length = self.model_state.transformer_state.context_length();

        if self.model_state.max_context_length.is_some_and(|max_context_length| context_length >= max_context_length) {
            self.decoding_state = DecodingState::Halted;
            return Ok(Some(
                prev_output
                    .resolve(
                        &mut self.model_state.tokens,
                        #[cfg(grammar)]
                        self.options.grammar.as_mut(),
                    )?
                    .0,
            ));
        }

        let capture_span = if let Some(capture_manager) = &self.model.engine.capture_manager
            && let Some(capture_request) = capture_manager.maybe_capture_decode_step()
        {
            prev_output.resolve(
                &mut self.model_state.tokens,
                #[cfg(grammar)]
                self.options.grammar.as_mut(),
            )?;
            Some(capture_request.start().map_err(LanguageModelStreamError::Backend)?)
        } else {
            None
        };

        let mut pending = Vec::new();
        let mut encoder = if let Some(encoder) = encoder {
            encoder
        } else {
            Encoder::<B>::new_with_pool(&self.model.engine.context, self.allocation_pool.clone())
                .map_err(LanguageModelStreamError::Backend)?
        };

        let speculation_batch = self
            .model_state
            .max_context_length
            .map_or(32, |max_context_length| 32.min(max_context_length - context_length));

        let (input_trie, chain_copy, full_accept) = if speculation_batch > 1
            && let Some(speculator) = &self.model.speculator
            && let (root_token, Some(output_norm)) = prev_output.resolve(
                &mut self.model_state.tokens,
                #[cfg(grammar)]
                self.options.grammar.as_mut(),
            )? {
            pending.push(encoder.end_encoding().submit());
            encoder = Encoder::<B>::new_with_pool(&self.model.engine.context, self.allocation_pool.clone())
                .map_err(LanguageModelStreamError::Backend)?;
            let trie = speculator.propose_tree(
                self.model_state.speculator_state.as_mut().unwrap(),
                output_norm,
                root_token as u32,
                self.model.decoder.embedding(),
                &self.model_state.prng,
                #[cfg(grammar)]
                self.options.grammar.as_mut(),
                DFlashTreeOptions {
                    budget: speculation_batch - 1,
                    frontier_width: 4,
                    children_per_node: 4,
                },
            )?;
            (trie, None, false)
        } else {
            let (token, chain_copy) = match &prev_output {
                ForwardPassChaining::Constant {
                    token,
                    ..
                } => (*token, None),
                ForwardPassChaining::InFlight(pending) => (0, Some(&pending.output_tokens)),
            };
            (TrieNode::new(token, self.model_state.prng.derive(context_length as u64)), chain_copy, true)
        };
        let input_flat_trie = input_trie.linearize();

        let token_ids = if let Some(chain_copy) = chain_copy {
            let mut token_ids =
                encoder.allocate_scratch(DataType::U32.size_in_bytes()).map_err(LanguageModelStreamError::Backend)?;
            encoder.encode_copy(chain_copy, .., &mut token_ids, ..);
            token_ids
        } else {
            let mut token_ids = encoder
                .allocate_constant(input_flat_trie.len() * DataType::U32.size_in_bytes())
                .map_err(LanguageModelStreamError::Backend)?;
            token_ids.copyin(&input_flat_trie.token_ids().map(|token_id| token_id as u32).collect::<Box<[u32]>>());
            token_ids
        };

        let input_flat_trie_nodes = input_flat_trie.token_subtrie_ranges().collect::<Box<[GpuTrieNode]>>();
        let batch_dim = BatchTopology::new(&input_flat_trie_nodes, full_accept);

        self.model_state
            .transformer_state
            .prepare(self.model_state.transformer_state.context_length(), batch_dim.size(), &self.model.engine.context)
            .map_err(LanguageModelStreamError::Backend)?;

        let hidden_feature_layer_indices =
            self.model.speculator.as_ref().map(|speculator| speculator.hidden_feature_layer_indices());

        let decoder_output = self.model.decoder.encode(
            &token_ids,
            &batch_dim,
            Some(0..batch_dim.size()),
            hidden_feature_layer_indices,
            &mut self.model_state.transformer_state,
            &mut encoder,
        )?;
        let logits = decoder_output.logits.unwrap();

        #[cfg(grammar)]
        let (bitmask, mut encoder) = if let Some(grammar) = self.options.grammar.as_mut() {
            if chain_copy.is_some() {
                pending.push(encoder.end_encoding().submit());

                let mut encoder = Encoder::<B>::new_with_pool(&self.model.engine.context, self.allocation_pool.clone())
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
        #[cfg(not(grammar))]
        let bitmask = None;

        let seeds = if matches!(self.options.sampling_method, SamplingMethod::Stochastic { .. }) {
            let mut seeds = encoder
                .allocate_constant(input_flat_trie.len() * DataType::U64.size_in_bytes())
                .map_err(LanguageModelStreamError::Backend)?;
            seeds.copyin(&input_flat_trie.token_seeds().collect::<Box<[u64]>>());
            Some(seeds)
        } else {
            None
        };

        let output_tokens = self
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

            if let Some(speculator) = self.model.speculator.as_ref() {
                let speculator_state = self.model_state.speculator_state.as_mut().unwrap();
                speculator
                    .append_state(
                        speculator_state,
                        decoder_output.hidden_features.as_ref().unwrap(),
                        &(0..batch_dim.size()).collect::<Box<[usize]>>(),
                        &mut encoder,
                    )
                    .map_err(LanguageModelStreamError::Backend)?;
            }

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
            _capture_span: capture_span,
            hidden_features: if full_accept {
                None
            } else {
                decoder_output.hidden_features
            },
            output_norm: decoder_output.final_hidden,
            output_tokens,
        });

        Ok(Some(
            prev_output
                .resolve(
                    &mut self.model_state.tokens,
                    #[cfg(grammar)]
                    self.options.grammar.as_mut(),
                )?
                .0,
        ))
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
                        Encoder::<B>::new_with_pool(&self.model.engine.context, self.allocation_pool.clone()).unwrap();
                    self.model_state.transformer_state.encode_accept(&[0], &mut encoder).unwrap();
                    if let Some(speculator) = self.model.speculator.as_ref() {
                        speculator
                            .append_state(
                                self.model_state.speculator_state.as_mut().unwrap(),
                                in_flight.hidden_features.as_deref().unwrap(),
                                &[0],
                                &mut encoder,
                            )
                            .unwrap();
                    }
                    encoder.end_encoding().submit().wait_until_completed().unwrap();
                }

                Some(in_flight.output_tokens.as_slice::<u32>()[0] as u64)
            },
            DecodingState::Accepting {
                full,
                num_accepted,
                hidden_features,
                output_norm: _,
            } => {
                assert!(num_accepted > 0 && num_accepted < full.len());

                let mut encoder =
                    Encoder::<B>::new_with_pool(&self.model.engine.context, self.allocation_pool.clone()).unwrap();
                let accepted_token_indicies =
                    full.iter().take(num_accepted + 1).map(|(i, _, _)| *i).collect::<Box<[usize]>>();
                self.model_state.transformer_state.encode_accept(&accepted_token_indicies, &mut encoder).unwrap();
                if let Some(speculator) = self.model.speculator.as_ref() {
                    speculator
                        .append_state(
                            self.model_state.speculator_state.as_mut().unwrap(),
                            hidden_features.as_deref().unwrap(),
                            &accepted_token_indicies,
                            &mut encoder,
                        )
                        .unwrap();
                }
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
