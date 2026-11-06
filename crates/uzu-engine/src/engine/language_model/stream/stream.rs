use std::{
    iter::{once, repeat_n},
    mem,
    range::Range,
    sync::Arc,
};

use futures::executor::block_on;
use shoji::traits::backend::chat_token::TokenStreamMetrics;

use crate::{
    backends::common::{
        Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable, Context, CpuBuffer,
        DeviceCapabilities, GlobalBufferCpuGpu, GpuBuffer,
        capture_matcher::{CAPTURE_MATCHER, CaptureMatcherSpan},
        gpu_types::trie::TrieNode as TrieNodeGpu,
        kernel::ContextRingUpdateKernel,
    },
    data_type::DataType,
    encodable_block::{batch_topology::BatchTopology, sampling::SamplingMethod},
    engine::language_model::{
        LanguageModel,
        state::LanguageModelState,
        stream::{LanguageModelStreamError, LanguageModelStreamOptions},
    },
    trie,
    utils::{
        size_for_shape::size_for_shape,
        timestamps::{record, wait},
    },
};

pub struct LanguageModelStream<'a, B: Backend> {
    submitter: StreamEnqueuer<'a, B>,
    /// Currently running job
    job: Option<GenerationJob<B>>,
    /// Resolved job from which we're emitting tokens
    emitting: Option<(ResolvedJob<B>, usize)>,
}

struct StreamEnqueuer<'a, B: Backend> {
    model: &'a LanguageModel<B>,
    model_state: &'a mut LanguageModelState<B>,
    options: LanguageModelStreamOptions,
    allocation_pool: Arc<B::AllocationPool>,
    context_ring: Option<B::GlobalGpuBuffer>,
    metrics: TokenStreamMetrics,
}

enum GenerationJob<B: Backend> {
    Pending {
        output_norm: Option<B::ScratchBuffer>,
        output_tokens: B::GlobalGpuBuffer,
        batch_dim: BatchTopology<B>,
        hidden_features: Option<Box<[B::ScratchBuffer]>>,
        capture_span: Option<CaptureMatcherSpan>,
        pending: <<B as Backend>::CommandBuffer as CommandBuffer>::Pending,
    },
    Resolved(ResolvedJob<B>),
    Invalid,
}

struct ResolvedJob<B: Backend> {
    output_tokens: Box<[u64]>,
    output_norm_buffer: Option<B::ScratchBuffer>,
    output_tokens_buffer: B::GlobalGpuBuffer,
    batch_dim: BatchTopology<B>,
    hidden_features: Option<Box<[B::ScratchBuffer]>>,
    accepted_indices_and_input_tokens: Option<(Box<[u32]>, Box<[u64]>)>,
    capture_span: Option<CaptureMatcherSpan>,
}

impl<B: Backend> GenerationJob<B> {
    fn batch_dim(&self) -> &BatchTopology<B> {
        match self {
            Self::Pending {
                batch_dim,
                ..
            } => batch_dim,
            Self::Resolved(resolved) => &resolved.batch_dim,
            Self::Invalid => unreachable!(),
        }
    }

    fn output_tokens_buffer(&self) -> &B::GlobalGpuBuffer {
        match self {
            Self::Pending {
                output_tokens,
                ..
            } => output_tokens,
            Self::Resolved(resolved) => &resolved.output_tokens_buffer,
            Self::Invalid => unreachable!(),
        }
    }

    fn output_norm_buffer(&self) -> Option<&B::ScratchBuffer> {
        match self {
            Self::Pending {
                output_norm,
                ..
            } => output_norm.as_ref(),
            Self::Resolved(resolved) => resolved.output_norm_buffer.as_ref(),
            Self::Invalid => unreachable!(),
        }
    }
}

impl<'a, B: Backend> StreamEnqueuer<'a, B> {
    fn new(
        model: &'a LanguageModel<B>,
        input: &[u64],
        model_state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<Self, LanguageModelStreamError<B>> {
        if model_state.tokens.is_empty() && input.is_empty() {
            return Err(LanguageModelStreamError::NoSeedToken);
        };

        if model_state
            .max_context_length
            .is_some_and(|max_context_length| model_state.tokens.len() + input.len() > max_context_length as usize)
        {
            return Err(LanguageModelStreamError::ContextOverflow);
        }

        let allocation_pool = model.engine.context.create_allocation_pool();

        let context_ring = if let Some(suffix_repetition_length) = options.sampling_method.suffix_repetition_length() {
            let mut context_ring = model
                .engine
                .context
                .create_buffer_cpu_gpu(size_for_shape(&[2 + suffix_repetition_length], DataType::U32))
                .map_err(LanguageModelStreamError::Backend)?;

            let state_tokens_range = Range::from(
                model_state.tokens.len().saturating_sub(suffix_repetition_length as usize)..model_state.tokens.len(),
            );

            context_ring.copyin(
                &once(0) // offset
                    .chain(once(state_tokens_range.iter().len() as u64)) // length
                    .chain(model_state.tokens[state_tokens_range].iter().copied()) // tokens
                    .chain(repeat_n(0, suffix_repetition_length as usize - state_tokens_range.iter().len())) // pad if not full
                    .map(|x| x as u32)
                    .collect::<Box<[_]>>(),
            );

            Some(context_ring.handoff())
        } else {
            None
        };

        let metrics = TokenStreamMetrics::default();

        Ok(StreamEnqueuer {
            model,
            model_state,
            options,
            allocation_pool,
            context_ring,
            metrics,
        })
    }

    async fn resolve<'b>(
        &mut self,
        job: &'b mut GenerationJob<B>,
    ) -> Result<&'b ResolvedJob<B>, LanguageModelStreamError<B>> {
        let owned = mem::replace(job, GenerationJob::Invalid);
        let resolved = match owned {
            GenerationJob::Pending {
                output_norm: output_norm_buffer,
                output_tokens: output_tokens_buffer,
                batch_dim,
                hidden_features,
                capture_span,
                pending,
            } => {
                wait(pending, self.options.timestamps.as_ref()).await.map_err(LanguageModelStreamError::Backend)?;
                let gpu_tokens = self
                    .model
                    .engine
                    .context
                    .copyout::<u32>(&output_tokens_buffer)
                    .await
                    .map_err(LanguageModelStreamError::Backend)?;

                let mut output_tokens = gpu_tokens.iter().map(|&t| t as u64).collect::<Box<[_]>>();

                let accepted_indices_and_input_tokens = if batch_dim.full_accept() {
                    if let Some(grammar) = self.options.grammar.as_mut() {
                        for &token_id in &output_tokens {
                            grammar.accept_token(token_id)?
                        }
                    }
                    None
                } else {
                    let input_tokens = self
                        .model
                        .engine
                        .context
                        .copyout::<u32>(batch_dim.token_ids().expect("trie batch carries token ids"))
                        .await
                        .map_err(LanguageModelStreamError::Backend)?;

                    let trie_nodes = self
                        .model
                        .engine
                        .context
                        .copyout::<TrieNodeGpu>(batch_dim.nodes().expect("trie batch carries trie nodes"))
                        .await
                        .map_err(LanguageModelStreamError::Backend)?;

                    let (accepted_token_indicies, accepted_input_token_ids, accepted_output_token_ids) =
                        trie::accept(&trie_nodes, &output_tokens, &input_tokens, self.options.grammar.as_mut())?;

                    output_tokens = accepted_output_token_ids;
                    Some((accepted_token_indicies, accepted_input_token_ids))
                };

                ResolvedJob {
                    output_tokens,
                    output_norm_buffer,
                    output_tokens_buffer,
                    batch_dim,
                    hidden_features,
                    accepted_indices_and_input_tokens,
                    capture_span,
                }
            },
            GenerationJob::Resolved(resolved) => resolved,
            GenerationJob::Invalid => unreachable!(),
        };

        *job = GenerationJob::Resolved(resolved);
        let GenerationJob::Resolved(resolved) = &*job else {
            unreachable!()
        };

        Ok(resolved)
    }

    async fn resolve_take(
        &mut self,
        mut job: GenerationJob<B>,
    ) -> Result<ResolvedJob<B>, LanguageModelStreamError<B>> {
        self.resolve(&mut job).await?;
        let GenerationJob::Resolved(resolved) = job else {
            unreachable!()
        };

        Ok(resolved)
    }

    pub fn prefill(
        &mut self,
        input: &[u64],
    ) -> Result<GenerationJob<B>, LanguageModelStreamError<B>> {
        let _capture_span = CAPTURE_MATCHER.capture::<B>(&self.model.engine.context, "prefill")?;

        // NOTE: this is required for attention correctness (hardcoded suffix
        // 1024). This is really bad design, attention should be rewritten to
        // allow on-demand suffix length
        let max_batch_size = 1024;
        let number_of_batches = input.len().div_ceil(max_batch_size);

        self.model_state
            .transformer_state
            .prepare(
                self.model_state.transformer_state.context_length() + ((number_of_batches - 1) * max_batch_size) as u32,
                usize::min(max_batch_size, input.len()) as u32,
            )
            .map_err(LanguageModelStreamError::Backend)?;

        let command_buffer = self
            .model
            .engine
            .context
            .create_command_buffer(Some("prefill"), Some(self.allocation_pool.clone()))
            .map_err(LanguageModelStreamError::Backend)?;
        let mut command_buffer = record(command_buffer, self.options.timestamps.as_ref());

        command_buffer.debug_point(&format!(
            "prefill at context {}, suffix {} ({} batches)",
            self.model_state.transformer_state.context_length(),
            input.len(),
            number_of_batches
        ));

        let hidden_feature_layer_indices =
            self.model.speculator.as_ref().map(|speculator| speculator.hidden_feature_layer_indices());

        let mut encode_batch = |input_chunk: &[u64], sample: bool| -> Result<_, LanguageModelStreamError<B>> {
            let token_ids = command_buffer
                .allocate_constant_from_slice(
                    &input_chunk.iter().map(|token_id| *token_id as u32).collect::<Box<[u32]>>(),
                )
                .map_err(LanguageModelStreamError::Backend)?;

            let batch_dim = BatchTopology::flat(input_chunk.len() as u32);

            let decoder_output = self.model.decoder.encode(
                &token_ids,
                &batch_dim,
                sample.then(|| (input_chunk.len() as u32 - 1..input_chunk.len() as u32).into()),
                hidden_feature_layer_indices,
                &mut self.model_state.transformer_state,
                &mut command_buffer,
            )?;

            self.model_state
                .transformer_state
                .encode_accept(&(0..input_chunk.len() as u32).collect::<Box<[u32]>>(), &mut command_buffer)
                .map_err(LanguageModelStreamError::Backend)?;

            if let Some(speculator) = self.model.speculator.as_ref() {
                let speculator_state = self.model_state.speculator_state.as_mut().unwrap();
                speculator.encode_accept(
                    speculator_state,
                    decoder_output.hidden_features.as_ref().unwrap().iter(),
                    &(0..input_chunk.len() as u32).collect::<Box<[u32]>>(),
                    &mut command_buffer,
                )?;
            }

            if !sample && let Some(suffix_repetition_length) = self.options.sampling_method.suffix_repetition_length() {
                self.model.context_ring_update.encode(
                    &token_ids,
                    self.context_ring.as_mut().unwrap(),
                    suffix_repetition_length,
                    input_chunk.len() as u32,
                    &mut command_buffer,
                );
            }

            self.model_state.tokens.extend(input_chunk);

            Ok((decoder_output, token_ids, batch_dim))
        };

        let mut token_chunks = input.chunks(max_batch_size);
        for input_chunk in token_chunks.by_ref().take(number_of_batches - 1) {
            encode_batch(input_chunk, false)?;
        }

        // `token_chunks` should have exactly one element left.
        let final_chunk = token_chunks.next().unwrap();

        let split_last_logits_row = self.model.decoder.prefill_cache_skips_trailing_layers();
        let (decoder_output, token_ids, batch_dim) = if split_last_logits_row && final_chunk.len() > 1 {
            let (start, end) = final_chunk.split_at(final_chunk.len() - 1);
            encode_batch(start, false)?;
            encode_batch(end, true)?
        } else {
            encode_batch(final_chunk, true)?
        };

        let logits = decoder_output.logits.unwrap();

        let seeds = if matches!(self.options.sampling_method, SamplingMethod::Stochastic { .. }) {
            Some(
                command_buffer
                    .allocate_constant_from_slice(&[self
                        .model_state
                        .prng
                        .derive(self.model_state.tokens.len() as u64 - 1)])
                    .map_err(LanguageModelStreamError::Backend)?,
            )
        } else {
            None
        };

        let bitmask = if let Some(grammar) = self.options.grammar.as_mut() {
            let mut bitmask = command_buffer
                .allocate_constant(
                    self.model.vocab_size.div_ceil(DataType::U32.size_in_bits()) * DataType::U32.size_in_bytes(),
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

        let output_norm = decoder_output.final_hidden;
        let sampled_row = batch_dim.size() - 1;
        let output_token = self
            .model
            .sampling
            .encode(
                &logits,
                seeds.as_ref(),
                bitmask.as_ref(),
                self.context_ring.as_ref(),
                Some(&token_ids),
                &batch_dim,
                (sampled_row..sampled_row + 1).into(),
                &self.options.sampling_method,
                &mut command_buffer,
            )
            .map_err(LanguageModelStreamError::Backend)?;

        if let Some(suffix_repetition_length) = self.options.sampling_method.suffix_repetition_length() {
            self.model.context_ring_update.encode(
                &token_ids,
                self.context_ring.as_mut().unwrap(),
                suffix_repetition_length,
                batch_dim.size(),
                &mut command_buffer,
            );
        }

        drop(seeds);
        drop(bitmask);
        drop(token_ids);

        self.metrics.num_prefill_forward_passes += 1;
        self.metrics.num_tokens_prefilled += input.len();
        self.metrics.num_tokens_proposed += 1;

        Ok(GenerationJob::Pending {
            output_norm,
            output_tokens: output_token,
            batch_dim,
            hidden_features: decoder_output.hidden_features,
            capture_span: None,
            pending: command_buffer.end_encoding().submit(),
        })
    }

    pub async fn get_and_queue_pass(
        &mut self,
        mut last_job: GenerationJob<B>,
    ) -> Result<(ResolvedJob<B>, GenerationJob<B>), LanguageModelStreamError<B>> {
        let mut capture_span = CAPTURE_MATCHER
            .capture::<B>(&self.model.engine.context, &format!("decode:{}", self.metrics.num_decode_forward_passes))?;

        let command_buffer = self
            .model
            .engine
            .context
            .create_command_buffer(
                Some(&format!("decode {}", self.metrics.num_decode_forward_passes)),
                Some(self.allocation_pool.clone()),
            )
            .map_err(LanguageModelStreamError::Backend)?;
        let mut command_buffer = record(command_buffer, self.options.timestamps.as_ref());

        // Speculator pass
        let context_length = self.model_state.transformer_state.context_length();
        let (batch_dim, trie_pending, flat_seed) = if let Some(speculator) = &self.model.speculator
            && let Some(shape) = speculator.make_shape(
                self.model_state.max_context_length.map(|max_context_length| max_context_length - context_length),
            ) {
            let last_batch_dim = last_job.batch_dim();

            let (target_output_norm, target_output_token) = if last_batch_dim.full_accept() {
                let norm_buffer = last_job.output_norm_buffer().unwrap();
                let token_buffer = last_job.output_tokens_buffer();
                (norm_buffer.subrange(..), token_buffer.subrange(..))
            } else {
                // TODO: CPU-bubble: for `accepted_indices`, to find where the root of the speculator should be.
                let resolved = self.resolve(&mut last_job).await?;
                let batch_dim = &resolved.batch_dim;
                let norm_buffer = resolved.output_norm_buffer.as_ref().unwrap();
                let token_buffer = &resolved.output_tokens_buffer;

                let (accepted_indices, _) = resolved.accepted_indices_and_input_tokens.as_ref().unwrap();

                let row = (*accepted_indices.last().unwrap()) as usize;
                let norm_row_bytes = norm_buffer.size() / batch_dim.size() as usize;
                let token_word_bytes = DataType::U32.size_in_bytes();
                (
                    norm_buffer.subrange(row * norm_row_bytes..(row + 1) * norm_row_bytes),
                    token_buffer.subrange(row * token_word_bytes..(row + 1) * token_word_bytes),
                )
            };

            let batch_dim = speculator.propose_tree(
                self.model_state.speculator_state.as_mut().unwrap(),
                target_output_norm,
                target_output_token,
                self.model.decoder.embedding(),
                shape,
                &self.options.sampling_method,
                &self.model_state.prng,
                &mut command_buffer,
            )?;

            let trie_pending = command_buffer.end_encoding().submit();
            command_buffer = record(
                self.model
                    .engine
                    .context
                    .create_command_buffer(
                        Some(&format!("decode {} (forward pass)", self.metrics.num_decode_forward_passes)),
                        Some(self.allocation_pool.clone()),
                    )
                    .map_err(LanguageModelStreamError::Backend)?,
                self.options.timestamps.as_ref(),
            );

            (batch_dim, Some(trie_pending), None)
        } else {
            let mut batch_dim = BatchTopology::flat(1);
            let context = command_buffer.context();
            let token_word_bytes = DataType::U32.size_in_bytes();
            *batch_dim.token_ids_mut() = Some(
                context
                    .create_buffer_gpu(batch_dim.size() as usize * token_word_bytes)
                    .map_err(LanguageModelStreamError::Backend)?,
            );

            let token_range = if last_job.batch_dim().full_accept() {
                0..token_word_bytes
            } else {
                // Technically a bubble, but only hit if at the very end of
                // context window.
                let resolved = self.resolve(&mut last_job).await?;
                let (accepted_indices, _) = resolved.accepted_indices_and_input_tokens.as_ref().unwrap();
                let row = (*accepted_indices.last().unwrap()) as usize;
                row * token_word_bytes..(row + 1) * token_word_bytes
            };

            command_buffer.encode_copy(
                last_job.output_tokens_buffer().subrange(token_range),
                batch_dim.token_ids_mut().as_mut().expect("decode batch carries token ids"),
            );

            let seed = if matches!(self.options.sampling_method, SamplingMethod::Stochastic { .. }) {
                Some(
                    command_buffer
                        .allocate_constant_from_slice(&[self.model_state.prng.derive(context_length as u64)])
                        .map_err(LanguageModelStreamError::Backend)?,
                )
            } else {
                None
            };

            (batch_dim, None, seed)
        };

        // Main (verifier) forward pass
        let token_ids = batch_dim.token_ids().expect("decode batch carries token ids");

        command_buffer.debug_point(&format!(
            "decode round {} at context {} suffix {}",
            self.metrics.num_decode_forward_passes,
            context_length,
            batch_dim.size()
        ));

        self.model_state
            .transformer_state
            .prepare(context_length, batch_dim.size())
            .map_err(LanguageModelStreamError::Backend)?;

        let hidden_feature_layer_indices =
            self.model.speculator.as_ref().map(|speculator| speculator.hidden_feature_layer_indices());

        let decoder_output = self.model.decoder.encode(
            token_ids,
            &batch_dim,
            Some((0..batch_dim.size()).into()),
            hidden_feature_layer_indices,
            &mut self.model_state.transformer_state,
            &mut command_buffer,
        )?;

        let bitmask = if let Some(grammar) = self.options.grammar.as_mut() {
            let mut bitmask = command_buffer
                .allocate_constant(
                    self.model.vocab_size.div_ceil(DataType::U32.size_in_bits())
                        * DataType::U32.size_in_bytes()
                        * batch_dim.size() as usize,
                )
                .map_err(LanguageModelStreamError::Backend)?;

            let was_filled = if let Some(trie_pending) = trie_pending {
                // Not a CPU-bubble, forward pass is overlapped.
                wait(trie_pending, self.options.timestamps.as_ref())
                    .await
                    .map_err(LanguageModelStreamError::Backend)?;

                let trie_nodes = self
                    .model
                    .engine
                    .context
                    .copyout::<TrieNodeGpu>(batch_dim.nodes().expect("trie decode carries trie nodes"))
                    .await
                    .map_err(LanguageModelStreamError::Backend)?;

                let trie_tokens = self
                    .model
                    .engine
                    .context
                    .copyout::<u32>(batch_dim.token_ids().expect("trie decode carries token ids"))
                    .await
                    .map_err(LanguageModelStreamError::Backend)?;

                let trie_tokens = trie_tokens.into_iter().map(|t| t as u64).collect::<Box<[_]>>();

                trie::fill_bitmasks(&trie_nodes, &trie_tokens, bitmask.as_slice_mut(), self.model.vocab_size, grammar)
            } else {
                grammar.next_bitmask(bitmask.as_slice_mut())
            };

            was_filled.then_some(bitmask)
        } else {
            None
        };

        let seeds = batch_dim
            .seeds()
            .map(|seeds| seeds as &dyn GpuBuffer<Backend = B>)
            .or_else(|| flat_seed.as_ref().map(|seed| seed as &dyn GpuBuffer<Backend = B>));

        let output_tokens = self
            .model
            .sampling
            .encode(
                &decoder_output.logits.unwrap(),
                seeds,
                bitmask.as_ref(),
                self.context_ring.as_ref(),
                Some(token_ids),
                &batch_dim,
                (0..batch_dim.size()).into(),
                &self.options.sampling_method,
                &mut command_buffer,
            )
            .map_err(LanguageModelStreamError::Backend)?;

        drop(bitmask);

        // new job and statistics
        self.metrics.num_decode_forward_passes += 1;
        self.metrics.num_tokens_proposed += batch_dim.size() as usize;

        if batch_dim.full_accept() {
            self.model_state
                .transformer_state
                .encode_accept(&(0..batch_dim.size()).collect::<Box<[u32]>>(), &mut command_buffer)
                .map_err(LanguageModelStreamError::Backend)?;

            if let Some(speculator) = self.model.speculator.as_ref() {
                let speculator_state = self.model_state.speculator_state.as_mut().unwrap();
                speculator.encode_accept(
                    speculator_state,
                    decoder_output.hidden_features.as_ref().unwrap().iter(),
                    &(0..batch_dim.size()).collect::<Box<[u32]>>(),
                    &mut command_buffer,
                )?;
            }

            if let Some(suffix_repetition_length) = self.options.sampling_method.suffix_repetition_length() {
                self.model.context_ring_update.encode(
                    &output_tokens,
                    self.context_ring.as_mut().unwrap(),
                    suffix_repetition_length,
                    batch_dim.size(),
                    &mut command_buffer,
                );
            }

            capture_span.take();
        }

        drop(flat_seed);

        let new_job = GenerationJob::Pending {
            output_norm: decoder_output.final_hidden,
            output_tokens,
            batch_dim,
            hidden_features: decoder_output.hidden_features,
            capture_span,
            pending: command_buffer.end_encoding().submit(),
        };

        // Return results
        // NOTE: not a CPU bubble. The next job has already been scheduled.
        let resolved_job = self.resolve_take(last_job).await?;

        Ok((resolved_job, new_job))
    }

    fn encode_accept(
        &mut self,
        resolved: &mut ResolvedJob<B>,
        emitted_len: usize,
    ) -> Result<Option<<<B as Backend>::CommandBuffer as CommandBuffer>::Pending>, LanguageModelStreamError<B>> {
        let command_buffer = self
            .model
            .engine
            .context
            .create_command_buffer(Some("accept"), Some(self.allocation_pool.clone()))
            .map_err(LanguageModelStreamError::Backend)?;
        let mut command_buffer = record(command_buffer, self.options.timestamps.as_ref());

        let batch_dim = &resolved.batch_dim;

        if batch_dim.full_accept() {
            return Ok(None);
        }

        let (accepted_token_indices, accepted_input_token_ids) =
            resolved.accepted_indices_and_input_tokens.as_ref().unwrap();
        let accepted_token_indices = &accepted_token_indices[..emitted_len];
        let accepted_input_token_ids = &accepted_input_token_ids[..emitted_len];

        self.model_state
            .transformer_state
            .encode_accept(accepted_token_indices, &mut command_buffer)
            .map_err(LanguageModelStreamError::Backend)?;

        if let Some(speculator) = self.model.speculator.as_ref() {
            speculator.encode_accept(
                self.model_state.speculator_state.as_mut().unwrap(),
                resolved.hidden_features.as_ref().unwrap().iter(),
                accepted_token_indices,
                &mut command_buffer,
            )?;
        }

        if let Some(suffix_repetition_length) = self.options.sampling_method.suffix_repetition_length() {
            // TODO: This can be provided by a `verify_trie` kernel.
            let accepted_input_token_ids_const = command_buffer
                .allocate_constant_from_slice(
                    &accepted_input_token_ids.iter().map(|token_id| *token_id as u32).collect::<Box<[u32]>>(),
                )
                .map_err(LanguageModelStreamError::Backend)?;

            self.model.context_ring_update.encode(
                &accepted_input_token_ids_const,
                self.context_ring.as_mut().unwrap(),
                suffix_repetition_length,
                accepted_token_indices.len() as u32,
                &mut command_buffer,
            );
        }

        resolved.capture_span.take();

        Ok(Some(command_buffer.end_encoding().submit()))
    }
}

impl<'a, B: Backend> LanguageModelStream<'a, B> {
    pub fn new(
        model: &'a LanguageModel<B>,
        input: &[u64],
        model_state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<Self, LanguageModelStreamError<B>> {
        let mut submitter = StreamEnqueuer::new(model, input, model_state, options)?;
        let job = submitter.prefill(input)?;

        Ok(Self {
            submitter,
            emitting: None,
            job: Some(job),
        })
    }

    pub fn closed(&self) -> bool {
        self.job.is_none() && self.emitting.is_none()
    }

    fn emit_token(&mut self) -> Option<u64> {
        let (resolved, i) = self.emitting.as_mut()?;
        if *i >= resolved.output_tokens.len() {
            return None;
        }

        let token = resolved.output_tokens[*i];
        *i += 1;

        self.submitter.model_state.tokens.push(token);
        self.submitter.metrics.num_tokens_accepted += 1;
        self.submitter.metrics.num_tokens_returned += 1;
        Some(token)
    }

    pub async fn generate(&mut self) -> Result<Option<u64>, LanguageModelStreamError<B>> {
        if let Some(token) = self.emit_token() {
            return Ok(Some(token));
        }

        if let Some(job) = self.job.take() {
            let context_full = self.submitter.model_state.max_context_length.is_some_and(|max_context_length| {
                self.submitter.model_state.transformer_state.context_length() >= max_context_length
            });

            if job.batch_dim().full_accept() && !context_full {
                let (last_job, new_job) = self.submitter.get_and_queue_pass(job).await?;
                self.job = Some(new_job);
                self.emitting = Some((last_job, 0));
            } else {
                let resolved = self.submitter.resolve_take(job).await?;
                self.emitting = Some((resolved, 0))
            }
        } else if let Some((mut queued, emitted_len)) = self.emitting.take() {
            let accept_pending = self.submitter.encode_accept(&mut queued, emitted_len)?;
            if let Some(pending) = accept_pending {
                wait(pending, self.submitter.options.timestamps.as_ref())
                    .await
                    .map_err(LanguageModelStreamError::Backend)?;
            };

            let context_full = self.submitter.model_state.max_context_length.is_some_and(|max_context_length| {
                self.submitter.model_state.transformer_state.context_length() >= max_context_length
            });

            if !context_full {
                // Important to keep `last_resolved` alive
                let (last_resolved, new_job) =
                    self.submitter.get_and_queue_pass(GenerationJob::Resolved(queued)).await?;
                let resolved_new = self.submitter.resolve_take(new_job).await?;
                drop(last_resolved);
                self.emitting = Some((resolved_new, 0))
            } else {
                assert!(self.emit_token().is_none());
            }
        }

        Ok(self.emit_token())
    }

    pub fn metrics(&self) -> &TokenStreamMetrics {
        &self.submitter.metrics
    }

    pub async fn close(&mut self) -> Result<(), LanguageModelStreamError<B>> {
        let Some((mut last_resolved, emitted_len)) = self.emitting.take() else {
            return Ok(());
        };

        let accept_pending = self.submitter.encode_accept(&mut last_resolved, emitted_len)?;
        if let Some(pending) = accept_pending {
            wait(pending, self.submitter.options.timestamps.as_ref())
                .await
                .map_err(LanguageModelStreamError::Backend)?;
        };

        if let Some(job) = self.job.take() {
            self.submitter.resolve_take(job).await?;
        }

        Ok(())
    }
}

impl<'a, B: Backend> Drop for LanguageModelStream<'a, B> {
    fn drop(&mut self) {
        if !self.closed()
            && self
                .submitter
                .model
                .engine
                .context
                .device_capabilities()
                .contains(DeviceCapabilities::GLOBAL_BUFFERS_CPU_ACCESSIBLE)
        {
            let _ = block_on(self.close());
        }
    }
}
