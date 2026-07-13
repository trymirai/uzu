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
    encodable_block::{
        batch_topology::BatchTopology,
        per_layer_embedding::{PREFILL_CHUNK_SIZE, PleLease, PleSession},
        sampling::SamplingMethod,
    },
    engine::language_model::{
        LanguageModel,
        grammar::Grammar,
        state::LanguageModelState,
        stream::{LanguageModelStreamError, LanguageModelStreamOptions},
    },
    trie::TrieNode,
};

enum ForwardPass<B: Backend> {
    Constant(u64),
    InFlight(DecodingStatePending<B>),
}

type PendingPasses<B> = Vec<(Pending<B>, Option<PleLease<B>>)>;

fn wait_pending<B: Backend>(passes: &mut PendingPasses<B>) -> Result<(), LanguageModelStreamError<B>> {
    let mut first_error = None;
    for (pending, ple_lease) in std::mem::take(passes) {
        let pending_error = pending.wait_until_completed().err().map(LanguageModelStreamError::Backend);
        let ple_error = ple_lease.and_then(|lease| lease.complete().err()).map(Into::into);
        first_error = first_error.or(pending_error).or(ple_error);
    }
    first_error.map_or(Ok(()), Err)
}

impl<B: Backend> ForwardPass<B> {
    fn resolve<'grammar>(
        &mut self,
        pending: &mut PendingPasses<B>,
        tokens: &mut Vec<u64>,
        grammar: Option<&mut (dyn Grammar + 'grammar)>,
    ) -> Result<u64, LanguageModelStreamError<B>> {
        match self {
            Self::Constant(token_id) => Ok(*token_id),
            Self::InFlight(in_flight) => {
                assert!(in_flight.full_accept);
                wait_pending(pending)?;
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
    output: Allocation<B>,
    ple_lease: Option<PleLease<B>>,
}

enum DecodingState<B: Backend> {
    Seeded(u64, Option<PleLease<B>>),
    ForwardPassPending(DecodingStatePending<B>),
    Accepting {
        full: Box<[(usize, u64, u64)]>,
        num_accepted: usize,
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
    options: LanguageModelStreamOptions<'a>,
    allocation_pool: Rc<AllocationPool<B>>,
    context_ring: Option<Allocation<B>>,
    decoding_state: DecodingState<B>,
    pending: PendingPasses<B>,
    next_pending: PendingPasses<B>,
    ple: PleSession<B>,
    metrics: TokenStreamMetrics,
}

impl<'a, B: Backend> Unpin for LanguageModelStream<'a, B> {}

impl<'a, B: Backend> LanguageModelStream<'a, B> {
    pub fn new(
        model: &'a LanguageModel<B>,
        input: &[u64],
        model_state: &'a mut LanguageModelState<B>,
        mut options: LanguageModelStreamOptions<'a>,
    ) -> Result<Self, LanguageModelStreamError<B>> {
        if model_state.poisoned {
            return Err(LanguageModelStreamError::StatePoisoned);
        }

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
        let staged_decode_supported = options.speculator.is_none()
            && options.grammar.is_none()
            && matches!(options.sampling_method, SamplingMethod::Greedy);
        let mut ple = PleSession::new(model.decoder.per_layer_embedding(), &model.context, staged_decode_supported)?;
        model_state.poisoned = true;
        let mut submitted_prefill_passes = Vec::new();
        let decoding_state = if !input.is_empty() {
            model_state.last_output_token.take();

            // NOTE: this is required for attention correctness (hardcoded suffix 1024). This is really bad design, attention should be rewritten to allow on-demand suffix length
            let max_batch_size = PREFILL_CHUNK_SIZE;
            let number_of_batches = input.len().div_ceil(max_batch_size);

            model_state
                .transformer_state
                .prepare(
                    model_state.transformer_state.context_length() + (number_of_batches - 1) * max_batch_size,
                    usize::min(max_batch_size, input.len()),
                    &model.context,
                )
                .map_err(LanguageModelStreamError::Backend)?;

            let split_logits_row = model.decoder.prefill_cache_skips_trailing_layers();
            let prefill_chunks = input
                .chunks(max_batch_size)
                .enumerate()
                .flat_map(|(batch_idx, input_chunk)| {
                    prefill_chunk_parts(input_chunk, batch_idx == number_of_batches - 1, split_logits_row)
                })
                .flatten()
                .collect::<Box<[_]>>();
            let split_prefill_chunks = ple.stages_prefill() && prefill_chunks.len() > 1;
            let prefill_pool = |last: bool| {
                last.then(|| allocation_pool.clone())
                    .unwrap_or_else(|| Rc::new(model.context.create_allocation_pool(false)))
            };
            let mut encoder = Some(
                Encoder::<B>::new_with_pool(&model.context, prefill_pool(!split_prefill_chunks))
                    .map_err(LanguageModelStreamError::Backend)?,
            );

            let (mut output, mut ple_lease, mut unsplit_prefill_lease) = (None, None, None);
            let mut staged_prefill = ple.stage_prefill(prefill_chunks[0].0)?;

            for (chunk_idx, &(input_chunk, sample_last)) in prefill_chunks.iter().enumerate() {
                if encoder.is_none() {
                    encoder = Some(
                        Encoder::<B>::new_with_pool(&model.context, prefill_pool(sample_last))
                            .map_err(LanguageModelStreamError::Backend)?,
                    );
                }
                let chunk_encoder = encoder.as_mut().expect("prefill encoder missing");

                let input_trie = TrieNode::flat(model_state.tokens.len(), input_chunk, &model_state.prng);
                let input_flat_trie = input_trie.linearize();

                let mut token_ids = chunk_encoder
                    .allocate_constant(input_chunk.len() * DataType::U64.size_in_bytes())
                    .map_err(LanguageModelStreamError::Backend)?;
                token_ids.copyin(input_chunk);

                let input_flat_trie_nodes = input_flat_trie.token_subtrie_ranges().collect::<Box<[GpuTrieNode]>>();
                let batch_dim = BatchTopology::new(&input_flat_trie_nodes, true);
                let prefill_lease = staged_prefill.take();

                let mut logits = model
                    .decoder
                    .encode(
                        &token_ids,
                        &batch_dim,
                        sample_last.then(|| (input_chunk.len() - 1)..input_chunk.len()),
                        &mut model_state.transformer_state,
                        chunk_encoder,
                        ple.source(prefill_lease.as_ref()),
                        &[],
                    )?
                    .logits;

                if sample_last {
                    let logits = logits.take().unwrap();

                    let seeds = if matches!(options.sampling_method, SamplingMethod::Stochastic { .. }) {
                        let mut seeds = chunk_encoder
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
                        let mut bitmask = chunk_encoder
                            .allocate_constant(
                                model.vocab_size.div_ceil(DataType::U32.size_in_bits()) * DataType::U32.size_in_bytes(),
                            )
                            .map_err(LanguageModelStreamError::Backend)?;

                        grammar.next_bitmask(bitmask.as_slice_mut()).then_some(bitmask)
                    } else {
                        None
                    };

                    let mut sample = ple.begin_sample()?;
                    let sampled_output = model
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
                            sample.readback(),
                            chunk_encoder,
                        )
                        .map_err(LanguageModelStreamError::Backend)?;
                    ple_lease = sample.submit(chunk_encoder)?;
                    output = Some(sampled_output);
                }

                model_state
                    .transformer_state
                    .encode_accept(&(0..input_chunk.len()).collect::<Box<[usize]>>(), chunk_encoder)
                    .map_err(LanguageModelStreamError::Backend)?;

                if let Some(suffix_repetition_length) = options.sampling_method.suffix_repetition_length() {
                    model.context_ring_update.encode(
                        &token_ids,
                        context_ring.as_mut().unwrap(),
                        suffix_repetition_length as u32,
                        input_chunk.len() as u32,
                        chunk_encoder,
                    );
                }

                drop(token_ids);
                drop(logits);
                model_state.tokens.extend(input_chunk);

                if split_prefill_chunks {
                    let chunk_encoder = encoder.take().expect("prefill encoder missing at submit");
                    submitted_prefill_passes.push((chunk_encoder.end_encoding().submit(), prefill_lease));
                    if !sample_last {
                        staged_prefill = ple.stage_prefill(prefill_chunks[chunk_idx + 1].0).map_err(|error| {
                            let _ = wait_pending(&mut submitted_prefill_passes);
                            error
                        })?;
                        wait_pending(&mut submitted_prefill_passes)?;
                    }
                } else {
                    unsplit_prefill_lease = prefill_lease;
                }
            }

            if let Some(encoder) = encoder {
                submitted_prefill_passes.push((encoder.end_encoding().submit(), unsplit_prefill_lease));
            }

            metrics.num_forward_passes += 1;
            metrics.num_tokens_prefilled += input.len();
            metrics.num_tokens_proposed += 1;
            metrics.num_tokens_accepted += 1;

            DecodingState::ForwardPassPending(DecodingStatePending {
                input_trie: TrieNode::new(0, 0),
                full_accept: true,
                output: output.unwrap(),
                ple_lease,
            })
        } else {
            // TODO: this leaks previous LanguageModelStreamOptions
            let seed_token = model_state.last_output_token.take().unwrap();
            let ple_lease = ple.stage_token(seed_token)?;
            DecodingState::Seeded(seed_token, ple_lease)
        };
        model_state.poisoned = false;

        Ok(LanguageModelStream {
            model,
            model_state,
            options,
            allocation_pool,
            context_ring,
            decoding_state,
            pending: submitted_prefill_passes,
            next_pending: Vec::new(),
            ple,
            metrics,
        })
    }

    fn generate(&mut self) -> Result<Option<u64>, LanguageModelStreamError<B>> {
        let (mut prev_output, encoder, mut current_ple_lease): (
            ForwardPass<B>,
            Option<Encoder<B>>,
            Option<PleLease<B>>,
        ) = match replace(&mut self.decoding_state, DecodingState::Invalid) {
            DecodingState::Seeded(seed_token, ple_lease) => {
                self.model_state.tokens.push(seed_token);
                if let Some(grammar) = self.options.grammar.as_deref_mut() {
                    let _ = grammar.accept_token(seed_token); // TODO: this should not be ignored
                }
                self.metrics.num_tokens_returned += 1;
                (ForwardPass::Constant(seed_token), None, ple_lease)
            },
            DecodingState::ForwardPassPending(mut forward_pass_pending) => {
                let ple_lease = forward_pass_pending.ple_lease.take();
                if forward_pass_pending.full_accept {
                    self.metrics.num_tokens_returned += 1;
                    (ForwardPass::InFlight(forward_pass_pending), None, ple_lease)
                } else {
                    wait_pending(&mut self.pending)?;
                    let sampled_tokens =
                        forward_pass_pending.output.as_slice::<u32>().iter().map(|x| *x as u64).collect::<Box<[u64]>>();
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
                    let mut encoder = Encoder::<B>::new_with_pool(&self.model.context, self.allocation_pool.clone())
                        .map_err(LanguageModelStreamError::Backend)?;
                    self.model_state
                        .transformer_state
                        .encode_accept(&accepted_token_indicies, &mut encoder)
                        .map_err(LanguageModelStreamError::Backend)?;
                    if let Some(suffix_repetition_length) = self.options.sampling_method.suffix_repetition_length() {
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
                    (ForwardPass::Constant(output_token_id), Some(encoder), None)
                }
            },
            DecodingState::Halted => return Ok(None),
            DecodingState::Invalid => unreachable!(),
        };

        let context_length = self.model_state.transformer_state.context_length();

        if self.model_state.max_context_length.is_some_and(|max_context_length| context_length >= max_context_length) {
            let token = prev_output.resolve(
                &mut self.pending,
                &mut self.model_state.tokens,
                self.options.grammar.as_deref_mut(),
            )?;
            self.decoding_state = DecodingState::Halted;
            return Ok(Some(token));
        }

        let mut encoder = if let Some(encoder) = encoder {
            encoder
        } else {
            Encoder::<B>::new_with_pool(&self.model.context, self.allocation_pool.clone())
                .map_err(LanguageModelStreamError::Backend)?
        };

        let (input_trie, chain_copy, full_accept) = if let Some(speculator) = &self.options.speculator {
            prev_output.resolve(
                &mut self.pending,
                &mut self.model_state.tokens,
                self.options.grammar.as_deref_mut(),
            )?;

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
                ForwardPass::Constant(token_id) => (*token_id, None),
                ForwardPass::InFlight(pending) => (0, Some(&pending.output)),
            };
            (TrieNode::new(token, self.model_state.prng.derive(context_length as u64)), chain_copy, true)
        };
        let input_flat_trie = input_trie.linearize();
        let copied_chain = chain_copy.is_some();
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

        let decoder_output = self.model.decoder.encode(
            &token_ids,
            &batch_dim,
            Some(0..batch_dim.size()),
            &mut self.model_state.transformer_state,
            &mut encoder,
            self.ple.source(current_ple_lease.as_ref()),
            &[],
        )?;

        let logits = decoder_output.logits.unwrap();

        let (bitmask, mut encoder) = if let Some(grammar) = self.options.grammar.as_deref_mut() {
            if copied_chain {
                self.next_pending.push((encoder.end_encoding().submit(), current_ple_lease.take()));

                let mut encoder = Encoder::<B>::new_with_pool(&self.model.context, self.allocation_pool.clone())
                    .map_err(LanguageModelStreamError::Backend)?;

                let mut bitmask = encoder
                    .allocate_constant(
                        self.model.vocab_size.div_ceil(DataType::U32.size_in_bits()) * DataType::U32.size_in_bytes(),
                    )
                    .map_err(LanguageModelStreamError::Backend)?;

                prev_output.resolve(&mut self.pending, &mut self.model_state.tokens, Some(grammar))?;
                (grammar.next_bitmask(bitmask.as_slice_mut()).then_some(bitmask), encoder)
            } else {
                let mut bitmasks = encoder
                    .allocate_constant(
                        input_flat_trie.len()
                            * self.model.vocab_size.div_ceil(DataType::U32.size_in_bits())
                            * DataType::U32.size_in_bytes(),
                    )
                    .map_err(LanguageModelStreamError::Backend)?;

                (
                    input_flat_trie
                        .fill_bitmasks(bitmasks.as_slice_mut(), self.model.vocab_size, grammar)
                        .then_some(bitmasks),
                    encoder,
                )
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

        let mut sample = self.ple.begin_sample()?;
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
                sample.readback(),
                &mut encoder,
            )
            .map_err(LanguageModelStreamError::Backend)?;
        let next_ple_lease = sample.submit(&mut encoder)?;
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

        self.next_pending.push((encoder.end_encoding().submit(), current_ple_lease));

        self.metrics.num_forward_passes += 1;
        self.metrics.num_tokens_proposed += input_flat_trie.len();
        if full_accept {
            self.metrics.num_tokens_accepted += input_flat_trie.len();
        }

        let token = prev_output.resolve(
            &mut self.pending,
            &mut self.model_state.tokens,
            self.options.grammar.as_deref_mut(),
        )?;
        self.pending = replace(&mut self.next_pending, Vec::new());
        self.decoding_state = DecodingState::ForwardPassPending(DecodingStatePending {
            input_trie,
            full_accept,
            output,
            ple_lease: next_ple_lease,
        });
        Ok(Some(token))
    }

    pub fn metrics(&self) -> &TokenStreamMetrics {
        &self.metrics
    }
}

impl<'a, B: Backend> Iterator for LanguageModelStream<'a, B> {
    type Item = Result<u64, LanguageModelStreamError<B>>;

    fn next(&mut self) -> Option<Result<u64, LanguageModelStreamError<B>>> {
        let result = self.generate();
        if result.is_err() {
            let _ = (wait_pending(&mut self.pending), wait_pending(&mut self.next_pending));
            self.model_state.poisoned = true;
            self.model_state.last_output_token = None;
            self.decoding_state = DecodingState::Halted;
        }
        result.transpose()
    }
}

impl<'a, B: Backend> LanguageModelStream<'a, B> {
    fn accept(
        &mut self,
        accepted: &[usize],
    ) -> Result<(), LanguageModelStreamError<B>> {
        let mut encoder = Encoder::<B>::new_with_pool(&self.model.context, self.allocation_pool.clone())
            .map_err(LanguageModelStreamError::Backend)?;
        self.model_state
            .transformer_state
            .encode_accept(accepted, &mut encoder)
            .map_err(LanguageModelStreamError::Backend)?;
        encoder.end_encoding().submit().wait_until_completed().map(|_| ()).map_err(LanguageModelStreamError::Backend)
    }

    pub fn finish(&mut self) -> Result<(), LanguageModelStreamError<B>> {
        if matches!(self.decoding_state, DecodingState::Invalid | DecodingState::Halted) {
            if matches!(self.decoding_state, DecodingState::Invalid) {
                self.model_state.poisoned = true;
                self.model_state.last_output_token = None;
                return Err(LanguageModelStreamError::StatePoisoned);
            }
            return Ok(());
        }
        let state = replace(&mut self.decoding_state, DecodingState::Invalid);
        let result = (|| {
            let last_output_token = match state {
                DecodingState::Seeded(seed_token, _) => Some(seed_token),
                DecodingState::ForwardPassPending(in_flight) => {
                    wait_pending(&mut self.pending).and(wait_pending(&mut self.next_pending))?;
                    if !in_flight.full_accept {
                        self.accept(&[0])?;
                    }
                    Some(in_flight.output.as_slice::<u32>()[0] as u64)
                },
                DecodingState::Accepting {
                    full,
                    num_accepted,
                } => {
                    assert!(num_accepted > 0 && num_accepted < full.len());
                    let accepted = full.iter().take(num_accepted + 1).map(|(i, _, _)| *i).collect::<Box<_>>();
                    self.accept(&accepted)?;
                    self.model_state.tokens.extend(full.iter().take(num_accepted).map(|(_, _, t)| *t));
                    Some(full[num_accepted].2)
                },
                DecodingState::Halted | DecodingState::Invalid => unreachable!(),
            };
            Ok(last_output_token)
        })();

        self.decoding_state = DecodingState::Halted;
        match &result {
            Ok(last_output_token) => self.model_state.last_output_token = *last_output_token,
            Err(_) => {
                self.model_state.poisoned = true;
                self.model_state.last_output_token = None;
            },
        }
        result.map(|_| ())
    }
}

impl<'a, B: Backend> Drop for LanguageModelStream<'a, B> {
    fn drop(&mut self) {
        if !matches!(self.decoding_state, DecodingState::Halted) {
            self.model_state.poisoned = true;
            self.model_state.last_output_token = None;
            let _ = (wait_pending(&mut self.pending), wait_pending(&mut self.next_pending));
            self.decoding_state = DecodingState::Halted;
        }
    }
}

unsafe impl<'a, B: Backend> Send for LanguageModelStream<'a, B> {} // TODO: this should be done properly
