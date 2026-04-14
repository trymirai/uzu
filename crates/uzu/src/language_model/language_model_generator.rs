use std::{
    any::Any,
    iter::repeat_n,
    ops::{Deref, DerefMut, Range},
    path::Path,
    time::Instant,
};

use itertools::{Either, Itertools, izip};

#[cfg(feature = "tracing")]
use super::trace_debug::TraceDebugLayerSnapshot;
use super::{
    gpu_capture::GpuCaptureManager,
    grammar::CompiledGrammar,
    kv_debug::{KvDebugLayerSnapshot, KvDebugSnapshot},
    language_model_generator_context::LanguageModelGeneratorContext,
    result::{GenerateResult, PrefillResult},
    rng::PRng,
    trace_debug::TraceDebugSnapshot,
};
use crate::{
    array::ArrayContextExt,
    backends::common::{
        Backend, Buffer, CommandBuffer, Context, Encoder, Executable,
        kernel::{TokenCopySampledKernel, TokenCopyToResultsKernel},
    },
    config::ModelMetadata,
    encodable_block::EncodingParameters,
    forward_pass::{
        cache_layers::{CacheLayer, CacheLayersSlice},
        kv_cache_layer::INVALID_POSITION,
        state::ForwardPassState,
    },
    session::{
        config::DecodingConfig,
        helpers::Context as LlmContext,
        parameter::{ConfigResolvableValue, ResolvableValue, SamplingMethod},
        types::Error,
    },
    trie::{TrieCreationConfig, TrieNode},
    utils::pointers::SendPtr,
};

#[derive(Debug, Clone)]
struct Task<'a> {
    token_ids: &'a [u64],
    token_subtrie_ranges: Option<&'a [[u32; 3]]>,
    token_positions: &'a [usize],
    token_bitmask: Option<&'a [u32]>,
    token_seeds: &'a [u64],
    expected_number_of_new_tokens: usize,
    active_row_count: usize,
    sampling_start: usize,
    sampling_length: usize,
    is_prefilling: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct TaskEncodingKey {
    context_len: usize,
    batch_size: usize,
    expected_number_of_new_tokens: usize,
    active_row_count: usize,
    sampling_method: SamplingMethod,
    sampling_start: usize,
    sampling_len: usize,
    has_bitmask: bool,
    is_prefilling: bool,
}

pub struct LanguageModelGenerator<B: Backend> {
    pub decoding_config: DecodingConfig,
    pub tokens: Vec<u64>,

    pub context: LanguageModelGeneratorContext<B>,
    pre_encoded_task: Option<(TaskEncodingKey, Executable<B>)>,
    registered_prefix_len: usize,
    gpu_capture: GpuCaptureManager<B>,
    last_trace_snapshot: Option<TraceDebugSnapshot>,
}

pub trait LanguageModelGeneratorTrait {
    fn prefill(
        &mut self,
        tokens: Vec<u64>,
        compiled_grammar: Option<&mut CompiledGrammar>,
        sampling_method: SamplingMethod,
        prefix_offset: usize,
        sample_suffix: bool,
    ) -> Result<PrefillResult, Error>;

    fn generate(
        &mut self,
        compiled_grammar: Option<&mut CompiledGrammar>,
        sampling_method: SamplingMethod,
    ) -> Result<GenerateResult, Error>;
    fn generate_from_token_path(
        &mut self,
        continuation: &[u64],
        compiled_grammar: Option<&mut CompiledGrammar>,
        sampling_method: SamplingMethod,
    ) -> Result<GenerateResult, Error>;

    fn prepare_async(
        &mut self,
        tokens_to_generate: usize,
    );
    fn async_generate(
        &mut self,
        pass_idx: usize,
        sampling_method: SamplingMethod,
        on_complete: Box<dyn FnOnce(u64) + Send>,
    ) -> Result<(), Error>;

    fn clear_cache(&mut self);
    fn reset_state(&mut self);
    fn peak_memory_usage(&self) -> Option<usize>;
    fn kv_storage_bytes(&self) -> usize;
    fn uses_materialized_transformer_state(&self) -> bool;
    fn kv_debug_snapshot(&self) -> KvDebugSnapshot;
    fn trace_debug_snapshot(&self) -> TraceDebugSnapshot;
    fn tokens(&self) -> &[u64];

    fn tokens_len(&self) -> usize;
    fn tokens_push(
        &mut self,
        token: u64,
    );
    fn generate_suffix_length(&self) -> usize;
    fn async_batch_size(
        &self,
        model_path: &Path,
    ) -> usize;

    fn get_slice(
        &self,
        range: Range<usize>,
    ) -> Option<Box<dyn Any>>;
    fn apply_slice(
        &mut self,
        slice: &dyn Any,
        range: Range<usize>,
    );

    fn build_llm_context(&self) -> Box<dyn Any>;
    fn reconfigure_from_context(
        &mut self,
        context: &dyn Any,
    );
}

impl<B: Backend> LanguageModelGeneratorTrait for LanguageModelGenerator<B> {
    fn prefill(
        &mut self,
        tokens: Vec<u64>,
        mut compiled_grammar: Option<&mut CompiledGrammar>,
        sampling_method: SamplingMethod,
        prefix_offset: usize,
        sample_suffix: bool,
    ) -> Result<PrefillResult, Error> {
        assert!(!tokens.is_empty());

        self.tokens.extend(tokens.clone());

        let tokens_length = tokens.len();

        let prefill_step_size = self.decoding_config.prefill_step_size.resolve(&self.context.model_config);
        let prefill_steps = tokens_length.div_ceil(prefill_step_size);
        let prefill_size = prefill_steps * prefill_step_size;

        let speculator = &self.decoding_config.speculator_config.speculator;

        let suffix_length = if sample_suffix {
            self.decoding_config.generate_suffix_length().saturating_sub(1).min(prefill_size - tokens_length)
        } else {
            prefill_size - tokens_length
        };
        let suffix_root = TrieNode::from_speculator(
            &tokens,
            &self.context.seed,
            compiled_grammar.as_deref_mut(),
            speculator.as_ref(),
            &TrieCreationConfig::default(),
            suffix_length + 1,
        );
        let flat_trie = suffix_root.linearize();

        let has_grammar = compiled_grammar.is_some();

        let token_ids =
            tokens.iter().copied().take(tokens_length - 1).chain(flat_trie.token_ids()).chunks(prefill_step_size);

        let token_subtrie_ranges = repeat_n(None, tokens_length - 1)
            .chain(flat_trie.token_subtrie_ranges().map(Some))
            .chunks(prefill_step_size);

        let token_positions = (prefix_offset..prefix_offset + tokens_length - 1)
            .chain(flat_trie.token_positions().map(|trie_position| prefix_offset + tokens_length - 1 + trie_position))
            .chunks(prefill_step_size);

        let single_token_bitmask_size = self.context.model_shape.bitmask_shape(1)[1];
        let token_bitmasks = repeat_n(None, tokens_length - 1).chain(flat_trie.token_masks()).chunks(prefill_step_size);

        let token_seeds = repeat_n(0, tokens_length - 1).chain(flat_trie.token_seeds()).chunks(prefill_step_size);

        let mut last_state: Option<ForwardPassState<B>> = None;
        let mut run_times: Vec<f64> = Vec::new();

        // Process each prefill step and update the KV cache.
        for (
            step,
            (step_token_ids, step_token_subtrie_ranges, step_token_positions, step_token_bitmasks, step_token_seeds),
        ) in izip!(&token_ids, &token_subtrie_ranges, &token_positions, &token_bitmasks, &token_seeds).enumerate()
        {
            let tokens_start_index = step * prefill_step_size;
            let tokens_end_index = tokens_start_index + prefill_step_size;

            let step_token_ids = step_token_ids.collect::<Box<[u64]>>();
            let step_token_subtrie_ranges = step_token_subtrie_ranges.collect::<Box<[Option<[u32; 3]>]>>();
            let step_token_subtrie_ranges: Option<Box<[[u32; 3]]>> =
                if let Some(trie_start) = step_token_subtrie_ranges.iter().position(|e| e.is_some()) {
                    Some(
                        step_token_subtrie_ranges
                            .iter()
                            .enumerate()
                            .map(|(i, me)| {
                                if let Some([subtrie_start, subtrie_end, height]) = me {
                                    [
                                        trie_start as u32 + subtrie_start,
                                        trie_start as u32 + subtrie_end,
                                        trie_start as u32 + height,
                                    ]
                                } else {
                                    [i as u32, step_token_subtrie_ranges.len() as u32 - 1, i as u32]
                                }
                            })
                            .collect(),
                    )
                } else {
                    None
                };
            let step_token_positions = step_token_positions.collect::<Box<[usize]>>();
            let step_token_seeds = step_token_seeds.collect::<Box<[u64]>>();

            let active_row_count = step_token_positions.len();
            let is_last_prefill_step = step == prefill_steps - 1;
            let should_sample_after_step = sample_suffix && is_last_prefill_step;

            // If we sample on the last prefill step, we only need logits/sampling
            // for tokens that are beyond the prompt prefix (i.e. starting at the
            // suffix-root token, which is the last prompt token).
            let (sampling_start, sampling_length) = if should_sample_after_step {
                let suffix_root_index_in_step = (tokens_length - 1).saturating_sub(tokens_start_index);
                let sampling_length = active_row_count.saturating_sub(suffix_root_index_in_step);
                debug_assert!(sampling_length > 0, "Expected at least one token to sample on the last prefill step");
                (suffix_root_index_in_step, sampling_length)
            } else {
                (0, 0)
            };

            let step_token_bitmask: Option<Box<[u32]>> = if has_grammar && sampling_length > 0 {
                Some(
                    step_token_bitmasks
                        .map(|mask| match mask {
                            Some(mask) => Either::Left(
                                mask.iter()
                                    .copied()
                                    .take(single_token_bitmask_size)
                                    .chain(repeat_n(0u32, single_token_bitmask_size.saturating_sub(mask.len()))),
                            ),
                            None => Either::Right(repeat_n(u32::MAX, single_token_bitmask_size)),
                        })
                        .flatten()
                        .collect::<Box<[u32]>>(),
                )
            } else {
                // Drain the chunk iterator to keep the other chunked iterators aligned.
                let _ = step_token_bitmasks.count();
                None
            };

            let should_capture = self.gpu_capture.should_capture_prefill(step == 0);

            if should_capture {
                let _ = self.gpu_capture.start_capture(&self.context.context, "prefill");
            }

            let _ = last_state.take();

            let task = Task {
                token_ids: &step_token_ids,
                token_subtrie_ranges: step_token_subtrie_ranges.as_deref(),
                token_positions: &step_token_positions,
                token_bitmask: step_token_bitmask.as_deref(),
                token_seeds: &step_token_seeds,
                expected_number_of_new_tokens: step_token_ids.len(),
                active_row_count,
                sampling_start,
                sampling_length,
                is_prefilling: !should_sample_after_step,
            };

            let allow_pre_encode = self.allow_pre_encode(task.token_ids.len());
            let (state, run_time) = self.run_model(task, allow_pre_encode, sampling_method)?;

            if should_capture {
                self.gpu_capture.stop_capture(&self.context.context, "prefill").map_err(|_| Error::CaptureFailed)?;
            }

            // Register the accepted prompt tokens from this step.
            let step_end_token_index = std::cmp::min(tokens_end_index, tokens_length);
            let mut tokens_processed_this_step = step_end_token_index - tokens_start_index;

            if step == prefill_steps - 1 && sample_suffix {
                tokens_processed_this_step = tokens_processed_this_step.saturating_sub(1);
            }

            if tokens_processed_this_step > 0 {
                self.update_cache_layers(&state, &(0..tokens_processed_this_step).collect::<Vec<usize>>(), None, true)?;

                self.context.cache_layers.borrow_mut().register_accepted_tokens(tokens_processed_this_step);

                self.registered_prefix_len = prefix_offset + tokens_start_index + tokens_processed_this_step;
            }

            last_state = Some(state);
            run_times.push(run_time);
        }

        let mut final_state = last_state.ok_or(Error::PrefillFailed)?;
        if !sample_suffix {
            self.sync_prefix();
            self.context.cache_layers.borrow_mut().reset_triattention_prune_counters();
            return Ok(PrefillResult {
                tokens: Vec::new(),
                forwardpass_durations: run_times,
            });
        }
        let sampled_tokens = self.read_sampling_output(&mut final_state)?;

        let last_suffix_start = prefill_step_size * (prefill_steps - 1);
        let suffix_root_index = (tokens_length - last_suffix_start) - 1;

        let (accepted_tokens, accepted_token_indices) =
            flat_trie.accept(&sampled_tokens, compiled_grammar.as_deref_mut());

        self.update_cache_layers(
            &final_state,
            &accepted_token_indices.into_iter().map(|p| suffix_root_index + p).collect::<Box<[usize]>>(),
            Some(last_suffix_start),
            false,
        )?;

        self.tokens.extend(accepted_tokens.clone());
        self.sync_prefix();
        self.context.cache_layers.borrow_mut().reset_triattention_prune_counters();

        Ok(PrefillResult {
            tokens: accepted_tokens,
            forwardpass_durations: run_times,
        })
    }

    fn generate(
        &mut self,
        mut compiled_grammar: Option<&mut CompiledGrammar>,
        sampling_method: SamplingMethod,
    ) -> Result<GenerateResult, Error> {
        let speculator = &self.decoding_config.speculator_config.speculator;

        let suffix_length = self.decoding_config.generate_suffix_length();
        let suffix_root = TrieNode::from_speculator(
            &self.tokens,
            &self.context.seed,
            compiled_grammar.as_deref_mut(),
            speculator.as_ref(),
            &TrieCreationConfig::default(),
            suffix_length,
        );
        self.generate_from_suffix_root(&suffix_root, compiled_grammar, sampling_method)
    }

    fn generate_from_token_path(
        &mut self,
        continuation: &[u64],
        compiled_grammar: Option<&mut CompiledGrammar>,
        sampling_method: SamplingMethod,
    ) -> Result<GenerateResult, Error> {
        LanguageModelGenerator::generate_from_token_path(self, continuation, compiled_grammar, sampling_method)
    }

    /// Prepares async buffers for generation.
    /// Must be called after prefill, before async_generate loop.
    fn prepare_async(
        &mut self,
        tokens_to_generate: usize,
    ) {
        let prefill_count = self.tokens.len();

        self.context.async_buffers.prepare_positions(prefill_count, tokens_to_generate);
        self.context.async_buffers.prepare_seeds(&self.context.seed, prefill_count, tokens_to_generate);
        self.context.async_buffers.reset_counter();
    }

    /// Submits a single async forward pass.
    /// Does NOT block (except when GPU capture is enabled for the first decode).
    ///
    /// - `pass_idx`: Index of this pass (0, 1, 2, ...)
    /// - `sampling_method`: Sampling configuration
    /// - `on_complete`: Callback receiving sampled token as u64
    fn async_generate(
        &mut self,
        pass_idx: usize,
        sampling_method: SamplingMethod,
        on_complete: Box<dyn FnOnce(u64) + Send>,
    ) -> Result<(), Error> {
        assert_eq!(self.decoding_config.generate_suffix_length(), 1, "async_generate only supports suffix_length=1");

        // Extract values from async_buffers before mutable borrow
        let current_counter = self.context.async_buffers.counter.get();
        let is_continuation = current_counter > 0;
        let batch_size = self.context.async_buffers.batch_size;
        let slot = pass_idx % batch_size;

        let results_buffer = self.context.async_buffers.results.clone();
        let async_positions_buffer = self.context.async_buffers.positions.clone();
        let async_seeds_buffer = self.context.async_buffers.seeds.clone();

        let last_token = *self.tokens.last().ok_or(Error::PrefillFailed)?;

        let token_position = unsafe {
            let ptr = async_positions_buffer.borrow().cpu_ptr().as_ptr() as *const u32;
            *ptr.add(pass_idx) as usize
        };

        let task = Task {
            token_ids: &[last_token],
            token_subtrie_ranges: None,
            token_positions: &[token_position],
            token_bitmask: None,
            token_seeds: &[0], // Ignored, using async buffer
            expected_number_of_new_tokens: 1,
            active_row_count: 1,
            sampling_start: 0,
            sampling_length: 1,
            is_prefilling: false,
        };

        let async_positions = Some((async_positions_buffer.clone(), pass_idx));
        let async_seeds = Some((async_seeds_buffer.clone(), pass_idx));

        let skip_token_ids_copy = pass_idx > 0;

        let is_first_decode = !is_continuation;
        let should_capture = self.gpu_capture.should_capture_decode(is_first_decode);
        if should_capture {
            let _ = self.gpu_capture.start_capture(&self.context.context, "decode");
        }

        let mut state = ForwardPassState::new_llm(
            self.context.context.clone(),
            &self.context.decoder_config,
            &self.context.model_shape,
            &self.context.scratch_buffers,
            self.context.cache_layers.clone(),
            self.context.shared_buffers.clone(),
            &task.token_ids,
            task.token_subtrie_ranges,
            &task.token_positions,
            task.token_bitmask,
            &task.token_seeds,
            task.active_row_count,
            /*sampling_start=*/ 0,
            /*sampling_length=*/ task.active_row_count,
            task.is_prefilling,
            skip_token_ids_copy,
            async_positions,
            async_seeds,
        );
        if let Some(sm) = state.sampling_method_mut() {
            *sm = Some(sampling_method);
        }

        let mut encoder = Encoder::<B>::new(self.context.context.as_ref())
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?;

        // Wait on previous pass if this is a continuation
        if is_continuation {
            encoder.encode_wait_for_event(&self.context.async_buffers.event, current_counter);
        }

        self.context
            .executables
            .encode(&mut state, &EncodingParameters::new(), &mut encoder)
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;

        // Encode sampling
        self.context.gpu_sampler.encode(&mut state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;

        // Copy sampled token: sampling_output → token_ids (for next pass)
        // and sampling_output → results[slot] (for callback)
        let sampling_output = state.sampling_output().expect("sampling_output must exist after sampling encode");
        let sampling_output_buf_rc = sampling_output.buffer();
        let sampling_output_buf_borrow = sampling_output_buf_rc.borrow();
        let token_ids_buf_rc = self.context.scratch_buffers.token_ids.buffer();
        let mut token_ids_buf_borrow = token_ids_buf_rc.borrow_mut();

        self.context.token_copy_sampled.encode(
            sampling_output_buf_borrow.deref(),
            token_ids_buf_borrow.deref_mut(),
            &mut encoder,
        );
        let results_offset = slot * std::mem::size_of::<u32>();
        self.context.token_copy_results.encode(
            sampling_output_buf_borrow.deref(),
            (results_buffer.borrow_mut().deref_mut(), results_offset),
            &mut encoder,
        );

        // Scatter + register for all transformer layers
        state.update_cache_after_acceptance(
            self.context.context.as_ref(),
            &[0],
            None,
            &mut encoder,
            &self.context.kv_cache_update,
        );
        self.context.cache_layers.borrow_mut().register_accepted_tokens(1);

        // Signal event for next pass
        let next_counter = current_counter + 1;
        encoder.encode_signal_event(&self.context.async_buffers.event, next_counter);
        self.context.async_buffers.counter.set(next_counter);

        // Add completion handler
        let results_buffer_ptr = SendPtr(results_buffer.borrow().cpu_ptr().as_ptr() as *const u32);

        let handler = move |result: Result<&<B::CommandBuffer as CommandBuffer>::Completed, B::Error>| {
            result.expect("async decoding forward pass completed with error");
            let token = { unsafe { *results_buffer_ptr.as_ptr().add(slot) as u64 } };
            on_complete(token);
        };

        encoder.add_completion_handler(handler);

        let pending = encoder.end_encoding().submit();

        if should_capture {
            pending.wait_until_completed().map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;
            self.gpu_capture.stop_capture(&self.context.context, "decode").map_err(|_| Error::CaptureFailed)?;
        }

        Ok(())
    }

    fn clear_cache(&mut self) {
        self.pre_encoded_task = None;
    }

    fn reset_state(&mut self) {
        self.context.cache_layers.borrow_mut().clear();
        self.tokens.clear();
        self.registered_prefix_len = 0;
        self.pre_encoded_task = None;
        self.last_trace_snapshot = None;
        self.gpu_capture.reset();

        let seed = self.decoding_config.sampling_seed.resolve();
        self.context.seed = PRng::new(seed);
        self.context.async_buffers.reset_counter();
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        self.context.context.peak_memory_usage()
    }
    fn kv_storage_bytes(&self) -> usize {
        self.context.cache_layers.borrow().kv_storage_bytes()
    }
    fn uses_materialized_transformer_state(&self) -> bool {
        self.context.cache_layers.borrow().uses_materialized_transformer_state()
    }
    fn kv_debug_snapshot(&self) -> KvDebugSnapshot {
        let cache_layers = self.context.cache_layers.borrow();
        let layers = cache_layers
            .data
            .iter()
            .enumerate()
            .filter_map(|(layer_index, layer)| {
                layer.as_transformer().map(|layer| self.snapshot_transformer_layer(layer_index, layer))
            })
            .collect();
        KvDebugSnapshot {
            layers,
        }
    }
    fn trace_debug_snapshot(&self) -> TraceDebugSnapshot {
        self.last_trace_snapshot.clone().expect("trace debug snapshot must be captured by the last forward pass")
    }
    fn tokens(&self) -> &[u64] {
        &self.tokens
    }

    fn tokens_len(&self) -> usize {
        self.tokens.len()
    }
    fn tokens_push(
        &mut self,
        token: u64,
    ) {
        self.tokens.push(token);
    }
    fn generate_suffix_length(&self) -> usize {
        self.decoding_config.generate_suffix_length()
    }
    fn async_batch_size(
        &self,
        model_path: &Path,
    ) -> usize {
        self.decoding_config.async_batch_size.resolve::<B>(model_path, self.context.context.as_ref())
    }

    fn get_slice(
        &self,
        range: Range<usize>,
    ) -> Option<Box<dyn Any>> {
        self.context.cache_layers.borrow().slice(&self.context.context, range).map(|s| Box::new(s) as Box<dyn Any>)
    }
    fn apply_slice(
        &mut self,
        slice: &dyn Any,
        range: Range<usize>,
    ) {
        let slice = slice.downcast_ref::<CacheLayersSlice<B>>().unwrap();
        self.context.cache_layers.borrow_mut().apply_slice(self.context.context.as_ref(), slice, Some(range));
    }

    fn build_llm_context(&self) -> Box<dyn Any> {
        let cache_layers = self.context.cache_layers.borrow().clone(&self.context.context);
        let context = LlmContext::new(self.tokens.clone(), cache_layers, self.decoding_config.clone());
        Box::new(context)
    }
    fn reconfigure_from_context(
        &mut self,
        context: &dyn Any,
    ) {
        let ctx = context.downcast_ref::<LlmContext<B>>().unwrap();
        let mut llm_state = self.context.cache_layers.borrow_mut();
        for (_layer_idx, (ctx_layer, gen_layer)) in
            ctx.cache_layers.data.iter().zip(llm_state.data.iter_mut()).enumerate()
        {
            match (ctx_layer, gen_layer) {
                (CacheLayer::Transformer(src), CacheLayer::Transformer(dst)) => {
                    let copy_rows = src.prefix_segment_length();
                    dst.state = src.state.clone();
                    dst.prefix_token_positions = src.prefix_token_positions.clone();
                    dst.next_token_position = src.next_token_position;
                    if let Some(triattention) = &mut dst.triattention {
                        triattention.tokens_since_last_prune = 0;
                    }

                    let mut dense_keys =
                        self.context.context.create_array(&dst.shape, dst.data_type, "reconfigure_keys");
                    let mut dense_values =
                        self.context.context.create_array(&dst.shape, dst.data_type, "reconfigure_values");
                    if copy_rows > 0 {
                        let src_keys = src.dense_keys().borrow();
                        let src_values = src.dense_values().borrow();
                        dense_keys.copy_slice(&src_keys, 1, 0..copy_rows, 0);
                        dense_values.copy_slice(&src_values, 1, 0..copy_rows, 0);
                    }
                    dst.compress_from(&dense_keys, &dense_values);
                },
                (CacheLayer::StateSpace(src), CacheLayer::StateSpace(dst)) => {
                    dst.conv_state.copy_from_array(&src.conv_state);
                    dst.ssm_state.copy_from_array(&src.ssm_state);
                },
                _ => panic!("Layer type mismatch when reconfiguring language model generator cache"),
            }
        }
        drop(llm_state);

        self.tokens = ctx.tokens.clone();
    }
}

impl<B: Backend> LanguageModelGenerator<B> {
    pub fn generate_from_token_path(
        &mut self,
        continuation: &[u64],
        mut compiled_grammar: Option<&mut CompiledGrammar>,
        sampling_method: SamplingMethod,
    ) -> Result<GenerateResult, Error> {
        let root_token = *self.tokens.last().ok_or(Error::PrefillFailed)?;
        let suffix_root = TrieNode::from_token_path(
            root_token,
            continuation,
            &self.context.seed,
            self.tokens.len() - 1,
            compiled_grammar.as_deref_mut(),
        );

        self.generate_from_suffix_root(&suffix_root, compiled_grammar, sampling_method)
    }

    pub fn generate_from_suffix_root(
        &mut self,
        suffix_root: &TrieNode,
        mut compiled_grammar: Option<&mut CompiledGrammar>,
        sampling_method: SamplingMethod,
    ) -> Result<GenerateResult, Error> {
        let flat_trie = suffix_root.linearize();
        let active_row_count = flat_trie.len();
        let suffix_length = self.decoding_config.generate_suffix_length().max(active_row_count);

        let token_ids =
            flat_trie.token_ids().chain(repeat_n(0, suffix_length - active_row_count)).collect::<Box<[u64]>>();

        let token_subtrie_ranges = flat_trie
            .token_subtrie_ranges()
            .chain(repeat_n([u32::MAX, u32::MAX, u32::MAX], suffix_length - active_row_count))
            .collect::<Box<[[u32; 3]]>>();

        let token_bitmask: Option<Box<[u32]>> = compiled_grammar.is_some().then(|| {
            let single_token_bitmask_size = self.context.model_shape.bitmask_shape(1)[1];
            flat_trie
                .token_masks()
                .chain(repeat_n(None, suffix_length - active_row_count))
                .map(|mask| match mask {
                    Some(mask) => Either::Left(
                        mask.iter()
                            .copied()
                            .take(single_token_bitmask_size)
                            .chain(repeat_n(0u32, single_token_bitmask_size.saturating_sub(mask.len()))),
                    ),
                    None => Either::Right(repeat_n(u32::MAX, single_token_bitmask_size)),
                })
                .flatten()
                .collect::<Box<[u32]>>()
        });

        let start_position = self.tokens.len() - 1;
        let token_positions = flat_trie
            .token_positions()
            .map(|trie_position| start_position + trie_position)
            .chain(repeat_n(INVALID_POSITION, suffix_length - active_row_count))
            .collect::<Box<[usize]>>();

        let token_seeds =
            flat_trie.token_seeds().chain(repeat_n(0, suffix_length - active_row_count)).collect::<Box<[u64]>>();

        let task = Task {
            token_ids: &token_ids,
            token_subtrie_ranges: Some(&token_subtrie_ranges),
            token_positions: &token_positions,
            token_bitmask: token_bitmask.as_deref(),
            token_seeds: &token_seeds,
            expected_number_of_new_tokens: 1,
            active_row_count,
            sampling_start: 0,
            sampling_length: active_row_count,
            is_prefilling: false,
        };

        let allow_pre_encode = self.allow_pre_encode(task.token_ids.len());
        let (mut state, run_time) = self.run_model(task, allow_pre_encode, sampling_method)?;

        let sampled_tokens = self.read_sampling_output(&mut state)?;
        let (accepted_tokens, accepted_token_indices) =
            flat_trie.accept(&sampled_tokens, compiled_grammar.as_deref_mut());
        let speculator_proposed = active_row_count.saturating_sub(1);
        let speculator_accepted = accepted_tokens.len().saturating_sub(1);

        self.update_cache_layers(&state, &accepted_token_indices, None, false)?;

        self.tokens.extend(accepted_tokens.clone());
        self.sync_prefix();
        self.prune_triattention_if_needed();

        Ok(GenerateResult {
            tokens: accepted_tokens,
            forwardpass_duration: run_time,
            speculator_proposed,
            speculator_accepted,
        })
    }

    fn snapshot_transformer_layer(
        &self,
        layer_index: usize,
        layer: &crate::forward_pass::kv_cache_layer::KVCacheLayer<B>,
    ) -> KvDebugLayerSnapshot {
        let (row_indices, positions) = match &layer.state {
            crate::forward_pass::kv_cache_layer::KVCacheLayerState::Full {
                prefix_len,
            } => ((0..*prefix_len).collect::<Vec<_>>(), layer.prefix_token_positions[..*prefix_len].to_vec()),
            crate::forward_pass::kv_cache_layer::KVCacheLayerState::Windowed {
                ring_offset,
                ring_length,
                window_length,
            } => {
                let row_indices =
                    (0..*ring_length).map(|offset| (ring_offset + offset) % window_length).collect::<Vec<_>>();
                let positions =
                    row_indices.iter().map(|&index| layer.prefix_token_positions[index]).collect::<Vec<_>>();
                (row_indices, positions)
            },
        };

        let mut keys = self.context.context.create_array(&layer.shape, layer.data_type, "kv_debug_keys");
        let mut values = self.context.context.create_array(&layer.shape, layer.data_type, "kv_debug_values");
        layer.materialize_into(&mut keys, &mut values);
        let (sparse_recent_positions, sparse_recent_values) =
            if let Some(recent_values) = &layer.sparse_value_recent_values {
                let sparse_value =
                    layer.sparse_value.as_ref().expect("SparseValue recent values require SparseValue state");
                let recent_len = positions.len().min(sparse_value.hot_value_capacity);
                let recent_start = positions.len() - recent_len;
                let recent_positions = positions[recent_start..].to_vec();
                let recent_row_indices = (recent_start..positions.len())
                    .map(|row_index| row_index % sparse_value.hot_value_capacity)
                    .collect::<Vec<_>>();
                (Some(recent_positions), Some(flatten_rows_as_f32(&recent_values.borrow(), &recent_row_indices)))
            } else {
                (None, None)
            };
        let sparse_pending_length = layer.sparse_value.as_ref().map_or(0, |state| state.pending_suffix_len);
        let sparse_pending_positions = (sparse_pending_length > 0).then(|| {
            assert!(
                sparse_pending_length <= positions.len(),
                "SparseValue pending rows must fit inside the captured prefix positions",
            );
            positions[positions.len() - sparse_pending_length..].to_vec()
        });
        let sparse_pending_values =
            layer.sparse_value_pending_values.as_ref().filter(|_| sparse_pending_length > 0).map(|pending_values| {
                flatten_rows_as_f32(&pending_values.borrow(), &(0..sparse_pending_length).collect::<Vec<_>>())
            });

        KvDebugLayerSnapshot {
            layer_index,
            positions,
            sparse_recent_positions,
            sparse_pending_positions,
            num_groups: layer.shape[0],
            head_dim: layer.shape[2],
            keys: flatten_rows_as_f32(&keys, &row_indices),
            values: flatten_rows_as_f32(&values, &row_indices),
            sparse_recent_values,
            sparse_pending_length,
            sparse_pending_values,
            storage_bytes: layer.storage_bytes(),
        }
    }

    pub fn new(
        model_path: &Path,
        decoding_config: DecodingConfig,
        model_metadata: &ModelMetadata,
    ) -> Result<Self, Error> {
        let gpu_capture = GpuCaptureManager::new();

        let context = LanguageModelGeneratorContext::new(model_path, &decoding_config, model_metadata)?;

        Ok(Self {
            decoding_config,
            tokens: Vec::new(),
            context,
            pre_encoded_task: None,
            registered_prefix_len: 0,
            gpu_capture,
            last_trace_snapshot: None,
        })
    }

    fn run_model(
        &mut self,
        task: Task,
        allow_pre_encode: bool,
        sampling_method: SamplingMethod,
    ) -> Result<(ForwardPassState<B>, f64), Error> {
        let run_start = Instant::now();

        let mut state = ForwardPassState::new_llm(
            self.context.context.clone(),
            &self.context.decoder_config,
            &self.context.model_shape,
            &self.context.scratch_buffers,
            self.context.cache_layers.clone(),
            self.context.shared_buffers.clone(),
            task.token_ids,
            task.token_subtrie_ranges,
            task.token_positions,
            task.token_bitmask,
            task.token_seeds,
            task.active_row_count,
            task.sampling_start,
            task.sampling_length,
            task.is_prefilling,
            false,
            None,
            None,
        );

        if let Some(method) = state.sampling_method_mut() {
            *method = Some(sampling_method);
        }

        let encoding_key = TaskEncodingKey {
            context_len: self.tokens.len(),
            batch_size: task.token_ids.len(),
            expected_number_of_new_tokens: task.expected_number_of_new_tokens,
            active_row_count: task.active_row_count,
            sampling_method,
            sampling_start: task.sampling_start,
            sampling_len: task.sampling_length,
            has_bitmask: task.token_bitmask.is_some(),
            is_prefilling: task.is_prefilling,
        };

        let is_first_decode = task.token_ids.len() == 1;
        let should_capture = self.gpu_capture.should_capture_decode(is_first_decode);

        if should_capture {
            self.gpu_capture.start_capture(&self.context.context, "decode").map_err(|_| Error::CaptureFailed)?;
            self.pre_encoded_task = None;
        }

        let sample = !task.is_prefilling;

        let executable = if let Some((pre_encoded_key, pre_encoded_executable)) = self.pre_encoded_task.take()
            && pre_encoded_key == encoding_key
        {
            pre_encoded_executable
        } else {
            self.encode_forward_pass(&mut state, &EncodingParameters::new(), sample)?
        };

        let pending = executable.submit();

        if allow_pre_encode {
            let mut next_encoding_key = encoding_key;

            next_encoding_key.context_len += 1;

            let next_executable =
                self.encode_forward_pass(&mut state, &EncodingParameters::new().with_projection(1), sample)?;

            self.pre_encoded_task = Some((next_encoding_key, next_executable));
        }

        pending.wait_until_completed().map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;
        #[cfg(feature = "tracing")]
        {
            self.last_trace_snapshot = Some(self.snapshot_trace_debug(&state));
        }

        let run_time = run_start.elapsed().as_secs_f64();

        if should_capture {
            self.gpu_capture.stop_capture(&self.context.context, "decode").map_err(|_| Error::CaptureFailed)?;
        }

        Ok((state, run_time))
    }

    fn encode_forward_pass(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters,
        sample: bool,
    ) -> Result<Executable<B>, Error> {
        let mut encoder = Encoder::<B>::new(self.context.context.as_ref())
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?;

        self.context
            .executables
            .encode(state, parameters, &mut encoder)
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;

        if sample {
            self.context.gpu_sampler.encode(state, &mut encoder).map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        }

        let executable = encoder.end_encoding();

        Ok(executable)
    }

    fn read_sampling_output(
        &mut self,
        state: &mut ForwardPassState<B>,
    ) -> Result<Vec<u64>, Error> {
        let sampling_output = state
            .sampling_output()
            .expect("Sampling output buffer not found - ensure sampling was encoded during forward pass");

        let output_view = sampling_output.as_view::<u32>();
        let batch_size = state.sampling_length();

        let mut result = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            result.push(output_view[[i]] as u64);
        }

        Ok(result)
    }

    fn update_cache_layers(
        &mut self,
        state: &ForwardPassState<B>,
        accepted_token_indices: &[usize],
        suffix_start: Option<usize>,
        wait_until_completed: bool,
    ) -> Result<(), Error> {
        let mut encoder = Encoder::<B>::new(self.context.context.as_ref())
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?;
        let must_wait =
            wait_until_completed || self.context.cache_layers.borrow().requires_synchronous_acceptance_update();

        state.update_cache_after_acceptance(
            self.context.context.as_ref(),
            accepted_token_indices,
            suffix_start,
            &mut encoder,
            &self.context.kv_cache_update,
        );

        let pending = encoder.end_encoding().submit();

        if must_wait {
            pending.wait_until_completed().map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;
        }
        Ok(())
    }

    fn allow_pre_encode(
        &self,
        batch_size: usize,
    ) -> bool {
        if !self.decoding_config.allow_pre_encode || self.context.context.debug_active() {
            return false;
        }

        let cache_layers = self.context.cache_layers.borrow();
        if !cache_layers.uses_materialized_transformer_state() {
            return true;
        }

        batch_size == 1 && !cache_layers.blocks_pre_encode_for_single_decode()
    }

    fn sync_prefix(&mut self) {
        if self.tokens.is_empty() {
            return;
        }

        let desired_prefix_len = self.tokens.len() - 1;
        if desired_prefix_len > self.registered_prefix_len {
            let number_of_accepted_tokens = desired_prefix_len - self.registered_prefix_len;
            self.context.cache_layers.borrow_mut().register_accepted_tokens(number_of_accepted_tokens);
            self.registered_prefix_len = desired_prefix_len;
        }
    }

    fn prune_triattention_if_needed(&mut self) {
        let has_triattention = self
            .context
            .cache_layers
            .borrow()
            .data
            .iter()
            .filter_map(|layer| layer.as_transformer())
            .any(|layer| layer.triattention.is_some());
        if !has_triattention {
            return;
        }

        let shared_buffers = self.context.shared_buffers.borrow();
        let global_rope = shared_buffers.global_rope.as_ref().expect("TriAttention requires global RoPE buffers");
        self.context.cache_layers.borrow_mut().prune_triattention_if_needed(&global_rope.cosines, &global_rope.sines);
    }

    #[cfg(feature = "tracing")]
    fn snapshot_trace_debug(
        &self,
        state: &ForwardPassState<B>,
    ) -> TraceDebugSnapshot {
        let active_rows = (0..state.active_row_count()).collect::<Vec<_>>();
        let traces = state.traces().borrow();
        let layers = traces
            .layer_results
            .iter()
            .enumerate()
            .map(|(layer_index, layer_traces)| {
                let layer_traces = layer_traces.borrow();
                TraceDebugLayerSnapshot {
                    layer_index,
                    model_dim: layer_traces.attention.shape()[1],
                    active_row_count: active_rows.len(),
                    sparse_value_single_decode_has_kv_cache: layer_traces.sparse_value_single_decode_has_kv_cache,
                    sparse_value_single_decode_has_sparse_value: layer_traces
                        .sparse_value_single_decode_has_sparse_value,
                    sparse_value_single_decode_suffix_length: layer_traces.sparse_value_single_decode_suffix_length,
                    sparse_value_single_decode_projection_step: layer_traces.sparse_value_single_decode_projection_step,
                    sparse_value_single_decode_is_trie: layer_traces.sparse_value_single_decode_is_trie,
                    sparse_value_single_decode_is_kv_cache_ring: layer_traces
                        .sparse_value_single_decode_is_kv_cache_ring,
                    attempted_sparse_value_single_decode: layer_traces.attempted_sparse_value_single_decode,
                    used_sparse_value_single_decode: layer_traces.used_sparse_value_single_decode,
                    pre_attention_norm: flatten_main_rows_as_f32(&layer_traces.pre_attention_norm, &active_rows),
                    attention: flatten_main_rows_as_f32(&layer_traces.attention, &active_rows),
                    sparse_expected_attention: layer_traces
                        .has_sparse_expected_attention
                        .then(|| flatten_main_rows_as_f32(&layer_traces.sparse_expected_attention, &active_rows)),
                    outputs: flatten_main_rows_as_f32(&layer_traces.outputs, &active_rows),
                }
            })
            .collect();
        TraceDebugSnapshot {
            layers,
        }
    }
}

fn flatten_rows_as_f32<B: Backend>(
    array: &crate::Array<B>,
    row_indices: &[usize],
) -> Vec<f32> {
    match array.data_type() {
        crate::DataType::BF16 => flatten_rows_as_f32_typed::<B, half::bf16>(array, row_indices),
        crate::DataType::F16 => flatten_rows_as_f32_typed::<B, half::f16>(array, row_indices),
        crate::DataType::F32 => flatten_rows_as_f32_typed::<B, f32>(array, row_indices),
        dtype => panic!("KV debug snapshot does not support dtype {dtype:?}"),
    }
}

fn flatten_rows_as_f32_typed<B: Backend, T>(
    array: &crate::Array<B>,
    row_indices: &[usize],
) -> Vec<f32>
where
    T: crate::ArrayElement + Copy,
    f32: From<T>,
{
    let shape = array.shape();
    let num_groups = shape[0];
    let sequence_length = shape[1];
    let head_dim = shape[2];
    let data = array.as_slice::<T>();
    let mut flat = Vec::with_capacity(num_groups * row_indices.len() * head_dim);

    for group_index in 0..num_groups {
        for &row_index in row_indices {
            assert!(row_index < sequence_length, "KV debug row index out of bounds");
            let start = (group_index * sequence_length + row_index) * head_dim;
            let end = start + head_dim;
            flat.extend(data[start..end].iter().copied().map(f32::from));
        }
    }

    flat
}

#[cfg(feature = "tracing")]
fn flatten_main_rows_as_f32<B: Backend>(
    array: &crate::Array<B>,
    row_indices: &[usize],
) -> Vec<f32> {
    match array.data_type() {
        crate::DataType::BF16 => flatten_main_rows_as_f32_typed::<B, half::bf16>(array, row_indices),
        crate::DataType::F16 => flatten_main_rows_as_f32_typed::<B, half::f16>(array, row_indices),
        crate::DataType::F32 => flatten_main_rows_as_f32_typed::<B, f32>(array, row_indices),
        dtype => panic!("trace debug snapshot does not support dtype {dtype:?}"),
    }
}

#[cfg(feature = "tracing")]
fn flatten_main_rows_as_f32_typed<B: Backend, T>(
    array: &crate::Array<B>,
    row_indices: &[usize],
) -> Vec<f32>
where
    T: crate::ArrayElement + Copy,
    f32: From<T>,
{
    let sequence_length = array.shape()[0];
    let model_dim = array.shape()[1];
    let data = array.as_slice::<T>();
    let mut flat = Vec::with_capacity(row_indices.len() * model_dim);
    for &row_index in row_indices {
        assert!(row_index < sequence_length, "trace debug row index out of bounds");
        let start = row_index * model_dim;
        let end = start + model_dim;
        flat.extend(data[start..end].iter().copied().map(f32::from));
    }
    flat
}
