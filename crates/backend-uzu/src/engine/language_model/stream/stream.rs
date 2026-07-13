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
        per_layer_embedding::{PleSource, PrefillChunkTicket, PrefillRing, RowRing, RowTicket, prefill_chunk_size},
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

enum ForwardPassChaining<B: Backend> {
    Constant {
        token_id: u64,
        row_ticket: Option<RowTicket<B>>,
    },
    InFlight(DecodingStatePending<B>),
}

struct PendingPass<B: Backend> {
    pending: Pending<B>,
    row_ticket: Option<RowTicket<B>>,
    prefill_ticket: Option<PrefillChunkTicket<B>>,
}

impl<B: Backend> PendingPass<B> {
    fn wait(self) -> Result<(), LanguageModelStreamError<B>> {
        let Self {
            pending,
            row_ticket,
            prefill_ticket,
        } = self;
        let mut first_error = pending.wait_until_completed().err().map(LanguageModelStreamError::Backend);
        if let Some(ticket) = row_ticket
            && let Err(error) = ticket.complete()
        {
            first_error.get_or_insert(error.into());
        }
        if let Some(ticket) = prefill_ticket
            && let Err(error) = ticket.complete()
        {
            first_error.get_or_insert(error.into());
        }
        first_error.map_or(Ok(()), Err)
    }

    fn drain(self) {
        let _ = self.wait();
    }
}

struct PendingPassDrain<B: Backend> {
    passes: Vec<PendingPass<B>>,
}

impl<B: Backend> PendingPassDrain<B> {
    fn new() -> Self {
        Self {
            passes: Vec::new(),
        }
    }

    fn push(
        &mut self,
        pending: PendingPass<B>,
    ) {
        self.passes.push(pending);
    }

    fn into_boxed_slice(mut self) -> Box<[PendingPass<B>]> {
        std::mem::take(&mut self.passes).into_boxed_slice()
    }
}

impl<B: Backend> Drop for PendingPassDrain<B> {
    fn drop(&mut self) {
        for pending in std::mem::take(&mut self.passes) {
            pending.drain();
        }
    }
}

impl<B: Backend> ForwardPassChaining<B> {
    fn row_ticket(&self) -> Option<&RowTicket<B>> {
        match self {
            Self::Constant {
                row_ticket,
                ..
            } => row_ticket.as_ref(),
            Self::InFlight(in_flight) => in_flight.row_ticket.as_ref(),
        }
    }

    fn take_row_ticket(&mut self) -> Option<RowTicket<B>> {
        match self {
            Self::Constant {
                row_ticket,
                ..
            } => row_ticket.take(),
            Self::InFlight(in_flight) => in_flight.row_ticket.take(),
        }
    }

    fn resolve<'grammar>(
        &mut self,
        tokens: &mut Vec<u64>,
        grammar: Option<&mut (dyn Grammar + 'grammar)>,
    ) -> Result<u64, LanguageModelStreamError<B>> {
        match self {
            Self::Constant {
                token_id,
                ..
            } => Ok(*token_id),
            Self::InFlight(in_flight) => {
                assert!(in_flight.full_accept);
                let mut first_error = None;
                for pending in replace(&mut in_flight.pending, Box::new([])) {
                    if let Err(error) = pending.wait() {
                        first_error.get_or_insert(error);
                    }
                }
                if let Some(error) = first_error {
                    return Err(error);
                }
                let output = in_flight.output.as_slice::<u32>();
                assert_eq!(output.len(), 1);
                let token_id = output[0] as u64;
                let row_ticket = in_flight.row_ticket.take();
                *self = Self::Constant {
                    token_id,
                    row_ticket,
                };
                tokens.push(token_id);
                if let Some(grammar) = grammar {
                    grammar.accept_token(token_id)?;
                }
                Ok(token_id)
            },
        }
    }
}

impl<B: Backend> Drop for ForwardPassChaining<B> {
    fn drop(&mut self) {
        match self {
            Self::Constant {
                row_ticket,
                ..
            } => {
                if let Some(ticket) = row_ticket.take() {
                    let _ = ticket.complete();
                }
            },
            Self::InFlight(in_flight) => {
                for pending in replace(&mut in_flight.pending, Box::new([])) {
                    pending.drain();
                }
                if let Some(ticket) = in_flight.row_ticket.take() {
                    let _ = ticket.complete();
                }
            },
        }
    }
}

struct DecodingStatePending<B: Backend> {
    input_trie: TrieNode,
    full_accept: bool,
    pending: Box<[PendingPass<B>]>,
    output: Allocation<B>,
    row_ticket: Option<RowTicket<B>>,
    _prefill_ring: Option<PrefillRing<B>>,
    _allocation_pool: Option<Rc<AllocationPool<B>>>,
}

enum DecodingState<B: Backend> {
    Seeded {
        seed_token: u64,
        row_ticket: Option<RowTicket<B>>,
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
    ple_row_ring: Option<RowRing<B>>,
    metrics: TokenStreamMetrics,
}

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
        let ring_mode_supported = options.speculator.is_none()
            && options.grammar.is_none()
            && matches!(options.sampling_method, SamplingMethod::Greedy);
        let mut ple_row_ring = if ring_mode_supported {
            model.decoder.create_ple_row_ring(&model.context)?
        } else {
            None
        };
        let mut submitted_prefill_passes = Vec::<PendingPass<B>>::new();
        let mut active_prefill_ring = None;
        let decoding_state_result = (|| -> Result<DecodingState<B>, LanguageModelStreamError<B>> {
            Ok(if !input.is_empty() {
                model_state.last_output_token.take();

                // NOTE: this is required for attention correctness (hardcoded suffix 1024). This is really bad design, attention should be rewritten to allow on-demand suffix length
                let max_batch_size = prefill_chunk_size();
                let number_of_batches = input.len().div_ceil(max_batch_size);
                active_prefill_ring = model.decoder.create_ple_prefill_ring(&model.context)?;

                model_state
                    .transformer_state
                    .prepare(
                        model_state.transformer_state.context_length() + (number_of_batches - 1) * max_batch_size,
                        usize::min(max_batch_size, input.len()),
                        &model.context,
                    )
                    .map_err(LanguageModelStreamError::Backend)?;

                let split_prefill_chunks = active_prefill_ring.is_some() && number_of_batches > 1;
                let mut encoder_pool = if active_prefill_ring.is_some() {
                    Rc::new(model.context.create_allocation_pool(false))
                } else {
                    allocation_pool.clone()
                };
                let mut encoder = Some(
                    Encoder::<B>::new_with_pool(&model.context, encoder_pool.clone())
                        .map_err(LanguageModelStreamError::Backend)?,
                );

                let mut output = None;
                let mut row_ticket = None;
                let mut unsplit_prefill_ticket = None;
                let mut retained_prefill_pool = None;
                let first_chunk_len = usize::min(max_batch_size, input.len());
                let mut staged_prefill_ticket =
                    active_prefill_ring.as_mut().map(|ring| ring.stage(0, &input[..first_chunk_len])).transpose()?;

                for (batch_idx, input_chunk) in input.chunks(max_batch_size).enumerate() {
                    if encoder.is_none() {
                        if active_prefill_ring.is_some() {
                            encoder_pool = Rc::new(model.context.create_allocation_pool(false));
                        }
                        encoder = Some(
                            Encoder::<B>::new_with_pool(&model.context, encoder_pool.clone())
                                .map_err(LanguageModelStreamError::Backend)?,
                        );
                    }
                    let last_batch = batch_idx == number_of_batches - 1;
                    if last_batch && active_prefill_ring.is_some() {
                        retained_prefill_pool = Some(encoder_pool.clone());
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
                    let prefill_ticket = staged_prefill_ticket.take();
                    let prepared_prefill = active_prefill_ring
                        .as_ref()
                        .zip(prefill_ticket.as_ref())
                        .map(|(ring, ticket)| ring.prepared(ticket));
                    let ple_source = if let Some(prepared_prefill) = prepared_prefill {
                        PleSource::StagedChunk(prepared_prefill)
                    } else if model.decoder.ple_rows_offloaded() {
                        PleSource::HostRows(input_chunk)
                    } else {
                        PleSource::Resident
                    };

                    let decoder_output = model.decoder.encode(
                        &token_ids,
                        &batch_dim,
                        last_batch.then(|| (input_chunk.len() - 1)..input_chunk.len()),
                        &mut model_state.transformer_state,
                        chunk_encoder,
                        ple_source,
                        &[],
                    )?;

                    let logits = decoder_output.logits;

                    if last_batch {
                        let logits = logits.unwrap();

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
                                    model.vocab_size.div_ceil(DataType::U32.size_in_bits())
                                        * DataType::U32.size_in_bytes(),
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
                                chunk_encoder,
                            )
                            .map_err(LanguageModelStreamError::Backend)?;
                        if let Some(row_ring) = &mut ple_row_ring {
                            row_ticket = Some(row_ring.publish_sample(&sampled_output, chunk_encoder)?);
                        }
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

                    model_state.tokens.extend(input_chunk);

                    if split_prefill_chunks {
                        let chunk_encoder = encoder.take().expect("prefill encoder missing at submit");
                        submitted_prefill_passes.push(PendingPass {
                            pending: chunk_encoder.end_encoding().submit(),
                            row_ticket: None,
                            prefill_ticket,
                        });
                        if active_prefill_ring.is_some() && !last_batch {
                            let next_start = (batch_idx + 1) * max_batch_size;
                            let next_end = usize::min(next_start + max_batch_size, input.len());
                            staged_prefill_ticket = active_prefill_ring
                                .as_mut()
                                .map(|ring| ring.stage(batch_idx + 1, &input[next_start..next_end]))
                                .transpose()?;
                            assert!(
                                submitted_prefill_passes.len() <= 2,
                                "staged prefill retained more than two pending passes"
                            );
                            for pending in std::mem::take(&mut submitted_prefill_passes) {
                                pending.wait()?;
                            }
                        }
                    } else {
                        unsplit_prefill_ticket = prefill_ticket;
                    }
                }

                if let Some(encoder) = encoder {
                    submitted_prefill_passes.push(PendingPass {
                        pending: encoder.end_encoding().submit(),
                        row_ticket: None,
                        prefill_ticket: unsplit_prefill_ticket,
                    });
                }
                let pending = std::mem::take(&mut submitted_prefill_passes).into_boxed_slice();

                metrics.num_forward_passes += 1;
                metrics.num_tokens_prefilled += input.len();
                metrics.num_tokens_proposed += 1;
                metrics.num_tokens_accepted += 1;

                DecodingState::ForwardPassPending(DecodingStatePending {
                    input_trie: TrieNode::new(0, 0),
                    full_accept: true,
                    pending,
                    output: output.unwrap(),
                    row_ticket,
                    _prefill_ring: active_prefill_ring.take(),
                    _allocation_pool: retained_prefill_pool,
                })
            } else {
                // TODO: this leaks previous LanguageModelStreamOptions
                let seed_token = model_state.last_output_token.take().unwrap();
                let row_ticket =
                    ple_row_ring.as_mut().map(|row_ring| row_ring.publish_known(seed_token)).transpose()?;
                DecodingState::Seeded {
                    seed_token,
                    row_ticket,
                }
            })
        })();
        let decoding_state = match decoding_state_result {
            Ok(decoding_state) => decoding_state,
            Err(error) => {
                for pending in std::mem::take(&mut submitted_prefill_passes) {
                    pending.drain();
                }
                drop(active_prefill_ring.take());
                model_state.poisoned = true;
                return Err(error);
            },
        };

        Ok(LanguageModelStream {
            model,
            model_state,
            options,
            allocation_pool,
            context_ring,
            decoding_state,
            ple_row_ring,
            metrics,
        })
    }

    fn halt_and_drain(&mut self) {
        match replace(&mut self.decoding_state, DecodingState::Halted) {
            DecodingState::Seeded {
                row_ticket: Some(ticket),
                ..
            } => {
                let _ = ticket.complete();
            },
            DecodingState::ForwardPassPending(in_flight) => {
                for pending in in_flight.pending {
                    pending.drain();
                }
                if let Some(ticket) = in_flight.row_ticket {
                    let _ = ticket.complete();
                }
            },
            DecodingState::Seeded {
                row_ticket: None,
                ..
            }
            | DecodingState::Accepting {
                ..
            }
            | DecodingState::Halted
            | DecodingState::Invalid => {},
        }
    }

    fn generate(&mut self) -> Result<Option<u64>, LanguageModelStreamError<B>> {
        let (mut prev_output, encoder): (ForwardPassChaining<B>, Option<Encoder<B>>) =
            match replace(&mut self.decoding_state, DecodingState::Invalid) {
                DecodingState::Seeded {
                    seed_token,
                    row_ticket,
                } => {
                    self.model_state.tokens.push(seed_token);
                    if let Some(grammar) = self.options.grammar.as_deref_mut() {
                        let _ = grammar.accept_token(seed_token); // TODO: this should not be ignored
                    }
                    self.metrics.num_tokens_returned += 1;
                    (
                        ForwardPassChaining::Constant {
                            token_id: seed_token,
                            row_ticket,
                        },
                        None,
                    )
                },
                DecodingState::ForwardPassPending(forward_pass_pending) => {
                    if forward_pass_pending.full_accept {
                        self.metrics.num_tokens_returned += 1;
                        (ForwardPassChaining::InFlight(forward_pass_pending), None)
                    } else {
                        let mut first_error = None;
                        for pending in forward_pass_pending.pending {
                            if let Err(error) = pending.wait() {
                                first_error.get_or_insert(error);
                            }
                        }
                        if let Some(error) = first_error {
                            return Err(error);
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
                        (
                            ForwardPassChaining::Constant {
                                token_id: output_token_id,
                                row_ticket: None,
                            },
                            Some(encoder),
                        )
                    }
                },
                DecodingState::Halted => return Ok(None),
                DecodingState::Invalid => unreachable!(),
            };

        let context_length = self.model_state.transformer_state.context_length();

        if self.model.decoder.ple_rows_offloaded() && self.options.speculator.is_none() && self.ple_row_ring.is_none() {
            prev_output.resolve(&mut self.model_state.tokens, self.options.grammar.as_deref_mut())?;
        }

        if self.model_state.max_context_length.is_some_and(|max_context_length| context_length >= max_context_length) {
            let token = prev_output.resolve(&mut self.model_state.tokens, self.options.grammar.as_deref_mut());
            let ticket = prev_output.take_row_ticket().map(|ticket| ticket.complete()).transpose();
            let token = token?;
            ticket?;
            self.decoding_state = DecodingState::Halted;
            return Ok(Some(token));
        }

        let mut pending = PendingPassDrain::<B>::new();
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
                ForwardPassChaining::Constant {
                    token_id,
                    ..
                } => (*token_id, None),
                ForwardPassChaining::InFlight(pending) => (0, Some(&pending.output)),
            };
            (TrieNode::new(token, self.model_state.prng.derive(context_length as u64)), chain_copy, true)
        };
        let input_flat_trie = input_trie.linearize();
        let row_token_ids = input_flat_trie.token_ids().collect::<Box<[u64]>>();
        let token_ids = if let Some(chain_copy) = chain_copy {
            let mut token_ids =
                encoder.allocate_scratch(DataType::U64.size_in_bytes()).map_err(LanguageModelStreamError::Backend)?;
            self.model.token_copy.encode(chain_copy, &mut token_ids, &mut encoder);
            token_ids
        } else {
            let mut token_ids = encoder
                .allocate_constant(input_flat_trie.len() * DataType::U64.size_in_bytes())
                .map_err(LanguageModelStreamError::Backend)?;
            token_ids.copyin(&row_token_ids);
            token_ids
        };

        let input_flat_trie_nodes = input_flat_trie.token_subtrie_ranges().collect::<Box<[GpuTrieNode]>>();
        let batch_dim = BatchTopology::new(&input_flat_trie_nodes, full_accept);

        self.model_state
            .transformer_state
            .prepare(self.model_state.transformer_state.context_length(), batch_dim.size(), &self.model.context)
            .map_err(LanguageModelStreamError::Backend)?;

        let prepared_row = self
            .ple_row_ring
            .as_ref()
            .zip(prev_output.row_ticket())
            .map(|(row_ring, ticket)| row_ring.prepared(ticket));
        let ple_source = if let Some(prepared_row) = prepared_row {
            PleSource::StagedRow(prepared_row)
        } else if self.model.decoder.ple_rows_offloaded() {
            PleSource::HostRows(row_token_ids.as_ref())
        } else {
            PleSource::Resident
        };
        let decoder_output = self.model.decoder.encode(
            &token_ids,
            &batch_dim,
            Some(0..batch_dim.size()),
            &mut self.model_state.transformer_state,
            &mut encoder,
            ple_source,
            &[],
        )?;

        let logits = decoder_output.logits.unwrap();

        let (bitmask, mut encoder) = if let Some(grammar) = self.options.grammar.as_deref_mut() {
            if chain_copy.is_some() {
                pending.push(PendingPass {
                    pending: encoder.end_encoding().submit(),
                    row_ticket: None,
                    prefill_ticket: None,
                });

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

        let next_row_ticket =
            self.ple_row_ring.as_mut().map(|row_ring| row_ring.publish_sample(&output, &mut encoder)).transpose()?;

        drop(token_ids);

        pending.push(PendingPass {
            pending: encoder.end_encoding().submit(),
            row_ticket: prev_output.take_row_ticket(),
            prefill_ticket: None,
        });

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
            row_ticket: next_row_ticket,
            _prefill_ring: None,
            _allocation_pool: None,
        });

        let token = prev_output.resolve(&mut self.model_state.tokens, self.options.grammar.as_deref_mut())?;
        Ok(Some(token))
    }

    pub fn metrics(&self) -> &TokenStreamMetrics {
        &self.metrics
    }
}

impl<'a, B: Backend> Iterator for LanguageModelStream<'a, B> {
    type Item = Result<u64, LanguageModelStreamError<B>>;

    fn next(&mut self) -> Option<Result<u64, LanguageModelStreamError<B>>> {
        match self.generate() {
            Ok(token) => token.map(Ok),
            Err(error) => {
                self.model_state.poisoned = true;
                self.halt_and_drain();
                Some(Err(error))
            },
        }
    }
}

impl<'a, B: Backend> LanguageModelStream<'a, B> {
    pub(crate) fn finish(&mut self) {
        if matches!(self.decoding_state, DecodingState::Invalid) {
            return;
        }
        let mut poisoned = self.model_state.poisoned;
        let last_output_token = match replace(&mut self.decoding_state, DecodingState::Invalid) {
            DecodingState::Seeded {
                seed_token,
                row_ticket,
            } => {
                if let Some(ticket) = row_ticket
                    && ticket.complete().is_err()
                {
                    poisoned = true;
                }
                Some(seed_token)
            },
            DecodingState::ForwardPassPending(in_flight) => {
                for pending in in_flight.pending {
                    if pending.wait().is_err() {
                        poisoned = true;
                    }
                }
                if let Some(row_ticket) = in_flight.row_ticket {
                    if row_ticket.complete().is_err() {
                        poisoned = true;
                    }
                }

                if !poisoned && !in_flight.full_accept {
                    match Encoder::<B>::new_with_pool(&self.model.context, self.allocation_pool.clone()) {
                        Ok(mut encoder) => {
                            if self.model_state.transformer_state.encode_accept(&[0], &mut encoder).is_err()
                                || encoder.end_encoding().submit().wait_until_completed().is_err()
                            {
                                poisoned = true;
                            }
                        },
                        Err(_) => poisoned = true,
                    }
                }

                (!poisoned).then(|| in_flight.output.as_slice::<u32>()[0] as u64)
            },
            DecodingState::Accepting {
                full,
                num_accepted,
            } => {
                assert!(num_accepted > 0 && num_accepted < full.len());

                match Encoder::<B>::new_with_pool(&self.model.context, self.allocation_pool.clone()) {
                    Ok(mut encoder) => {
                        let accepted = full.iter().take(num_accepted + 1).map(|(i, _, _)| *i).collect::<Box<_>>();
                        if self.model_state.transformer_state.encode_accept(&accepted, &mut encoder).is_err()
                            || encoder.end_encoding().submit().wait_until_completed().is_err()
                        {
                            poisoned = true;
                            None
                        } else {
                            self.model_state.tokens.extend(full.iter().take(num_accepted).map(|(_, _, t)| *t));
                            Some(full[num_accepted].2)
                        }
                    },
                    Err(_) => {
                        poisoned = true;
                        None
                    },
                }
            },
            DecodingState::Halted => None,
            DecodingState::Invalid => None, // TODO: proper error handling
        };

        self.model_state.poisoned = poisoned;
        self.model_state.last_output_token = (!poisoned).then_some(last_output_token).flatten();
    }
}

impl<'a, B: Backend> Drop for LanguageModelStream<'a, B> {
    fn drop(&mut self) {
        self.finish();
    }
}

unsafe impl<'a, B: Backend> Send for LanguageModelStream<'a, B> {} // TODO: this should be done properly
