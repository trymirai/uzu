use std::{
    any::Any,
    collections::HashMap,
    iter::repeat_n,
    ops::{Deref, DerefMut, Range},
    path::Path,
    time::Instant,
};

use itertools::{Either, Itertools, izip};

use super::{
    gpu_capture::GpuCaptureManager,
    grammar::CompiledGrammar,
    language_model_generator_context::LanguageModelGeneratorContext,
    result::{GenerateResult, PrefillResult},
    rng::PRng,
};
use crate::{
    backends::common::{
        Backend, Buffer, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial,
        CommandBufferPending, Context,
        kernel::{MaskUpdateKernel, TokenCopySampledKernel, TokenCopyToResultsKernel},
    },
    encodable_block::{EncodableBlock, EncodingParameters},
    forward_pass::{
        cache_layers::{CacheLayer, CacheLayersSlice},
        kv_cache_layer::{AttentionBiasUpdate, INVALID_POSITION},
        state::ForwardPassState,
    },
    session::{
        config::DecodingConfig,
        helpers::Context as LlmContext,
        parameter::{ConfigResolvableValue, ResolvableValue, SamplingMethod},
        types::Error,
    },
    trie::TrieNode,
};

#[derive(Debug, Clone)]
struct Task<'a> {
    token_ids: &'a [u64],
    token_positions: &'a [usize],
    token_bitmask: Option<&'a [u32]>,
    token_seeds: &'a [u64],
    expected_number_of_new_tokens: usize,
    active_suffix_length: usize,
    sampling_start: usize,
    sampling_length: usize,
    is_prefilling: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct TaskEncodingKey {
    context_len: usize,
    batch_size: usize,
    expected_number_of_new_tokens: usize,
    active_suffix_len: usize,
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
    pre_encoded_task: Option<(TaskEncodingKey, <B::CommandBuffer as CommandBuffer>::Executable)>,
    registered_prefix_len: usize,
    gpu_capture: GpuCaptureManager<B>,
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
            self.decoding_config.generate_suffix_length().saturating_sub(1)
        } else {
            prefill_size - tokens_length
        };
        let suffix_root = TrieNode::from_speculator(
            &tokens,
            &self.context.seed,
            compiled_grammar.as_deref_mut(),
            speculator.as_ref(),
            &speculator.trie_creation_config(),
            suffix_length + 1,
        );
        let flat_trie = suffix_root.linearize();

        let has_grammar = compiled_grammar.is_some();

        let token_ids =
            tokens.iter().copied().take(tokens_length - 1).chain(flat_trie.token_ids()).chunks(prefill_step_size);

        let token_positions = (prefix_offset..prefix_offset + tokens_length - 1)
            .chain(flat_trie.token_positions().map(|trie_position| prefix_offset + tokens_length - 1 + trie_position))
            .chunks(prefill_step_size);

        let single_token_bitmask_size = self.context.model_shape.bitmask_shape(1)[1];
        let token_bitmasks = repeat_n(None, tokens_length - 1).chain(flat_trie.token_masks()).chunks(prefill_step_size);

        let token_seeds = repeat_n(0, tokens_length - 1).chain(flat_trie.token_seeds()).chunks(prefill_step_size);

        let mut last_state: Option<ForwardPassState<B>> = None;
        let mut run_times: Vec<f64> = Vec::new();

        // Process each prefill step and update the KV cache.
        for (step, (step_token_ids, step_token_positions, step_token_bitmasks, step_token_seeds)) in
            izip!(&token_ids, &token_positions, &token_bitmasks, &token_seeds).enumerate()
        {
            let tokens_start_index = step * prefill_step_size;
            let tokens_end_index = tokens_start_index + prefill_step_size;

            let step_token_ids = step_token_ids.collect::<Box<[u64]>>();
            let step_token_positions = step_token_positions.collect::<Box<[usize]>>();
            let step_token_seeds = step_token_seeds.collect::<Box<[u64]>>();

            let active_suffix_length = step_token_positions.len();
            let is_last_prefill_step = step == prefill_steps - 1;
            let should_sample_after_step = sample_suffix && is_last_prefill_step;

            // If we sample on the last prefill step, we only need logits/sampling
            // for tokens that are beyond the prompt prefix (i.e. starting at the
            // suffix-root token, which is the last prompt token).
            let (sampling_start, sampling_length) = if should_sample_after_step {
                let suffix_root_index_in_step = (tokens_length - 1).saturating_sub(tokens_start_index);
                let sampling_length = active_suffix_length.saturating_sub(suffix_root_index_in_step);
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

            objc2::rc::autoreleasepool(|_pool| {
                let _ = last_state.take();
            });

            let task = Task {
                token_ids: &step_token_ids,
                token_positions: &step_token_positions,
                token_bitmask: step_token_bitmask.as_deref(),
                token_seeds: &step_token_seeds,
                expected_number_of_new_tokens: step_token_ids.len(),
                active_suffix_length,
                sampling_start,
                sampling_length,
                is_prefilling: !should_sample_after_step,
            };

            let (state, run_time) = self.run_model(
                task,
                false,
                self.allow_pre_encode(),
                sampling_method,
                self.should_fill_attention_bias(),
            )?;

            if should_capture {
                self.gpu_capture.stop_capture(&self.context.context, "prefill").map_err(|_| Error::CaptureFailed)?;
            }

            // Register the accepted prompt tokens from this step.
            let step_end_token_index = std::cmp::min(tokens_end_index, tokens_length);
            let tokens_processed_this_step = step_end_token_index - tokens_start_index;

            if tokens_processed_this_step > 0 {
                let mut positions_for_step: Vec<usize> =
                    (tokens_start_index..step_end_token_index).map(|idx| idx + prefix_offset).collect();
                if step == prefill_steps - 1 && sample_suffix {
                    // Exclude the last token because it belongs to the suffix for sampling.
                    positions_for_step.pop();
                }

                if !positions_for_step.is_empty() {
                    let accept_indices_for_step: Vec<usize> = (0..positions_for_step.len()).collect();
                    if !accept_indices_for_step.is_empty() {
                        self.update_cache_layers(&accept_indices_for_step, None, true)?;
                    }

                    self.context.cache_layers.borrow_mut().register_accepted_tokens(&positions_for_step);

                    if let Some(&last_idx) = positions_for_step.last() {
                        self.registered_prefix_len = last_idx + 1;
                    }
                }
            }

            last_state = Some(state);
            run_times.push(run_time);
        }

        let mut final_state = last_state.ok_or(Error::PrefillFailed)?;
        if !sample_suffix {
            self.sync_prefix();
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
            &accepted_token_indices.into_iter().map(|p| suffix_root_index + p).collect::<Box<[usize]>>(),
            Some(last_suffix_start),
            false,
        )?;

        self.tokens.extend(accepted_tokens.clone());
        self.sync_prefix();

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
            &speculator.trie_creation_config(),
            suffix_length,
        );

        let flat_trie = suffix_root.linearize();
        let active_suffix_length = flat_trie.len();

        let token_ids =
            flat_trie.token_ids().chain(repeat_n(0, suffix_length - active_suffix_length)).collect::<Box<[u64]>>();

        let token_bitmask: Option<Box<[u32]>> = compiled_grammar.is_some().then(|| {
            let single_token_bitmask_size = self.context.model_shape.bitmask_shape(1)[1];
            flat_trie
                .token_masks()
                .chain(repeat_n(None, suffix_length - active_suffix_length))
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
            .chain(repeat_n(INVALID_POSITION, suffix_length - active_suffix_length))
            .collect::<Box<[usize]>>();

        let token_seeds =
            flat_trie.token_seeds().chain(repeat_n(0, suffix_length - active_suffix_length)).collect::<Box<[u64]>>();

        let task = Task {
            token_ids: &token_ids,
            token_positions: &token_positions,
            token_bitmask: token_bitmask.as_deref(),
            token_seeds: &token_seeds,
            expected_number_of_new_tokens: 1,
            active_suffix_length,
            sampling_start: 0,
            sampling_length: active_suffix_length,
            is_prefilling: false,
        };

        let (mut state, run_time) =
            self.run_model(task, false, self.allow_pre_encode(), sampling_method, self.should_fill_attention_bias())?;

        let sampled_tokens = self.read_sampling_output(&mut state)?;

        let (accepted_tokens, accepted_token_indices) =
            flat_trie.accept(&sampled_tokens, compiled_grammar.as_deref_mut());
        let speculator_proposed = active_suffix_length.saturating_sub(1);
        let speculator_accepted = accepted_tokens.len().saturating_sub(1);

        self.update_cache_layers(&accepted_token_indices, None, false)?;

        self.tokens.extend(accepted_tokens.clone());
        self.sync_prefix();

        Ok(GenerateResult {
            tokens: accepted_tokens,
            forwardpass_duration: run_time,
            speculator_proposed,
            speculator_accepted,
        })
    }

    /// Prepares async buffers for generation.
    /// Must be called after prefill, before async_generate loop.
    fn prepare_async(
        &mut self,
        tokens_to_generate: usize,
    ) {
        let prefill_count = self.tokens.len();
        let first_decode_position = prefill_count.saturating_sub(1);

        // Fill attention bias from KV cache state for the first decode position.
        // Windowed buffers get the correct mask; GPU patches maintain it for subsequent passes.
        // Full-attention buffers also get filled (masking beyond-prefix positions).
        self.context.cache_layers.borrow().fill_attention_bias_scratch(
            &self.context.scratch_buffers.attention_window_size_to_bias,
            &[first_decode_position],
            1,
            &self.context.context,
        );

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
            token_positions: &[token_position],
            token_bitmask: None,
            token_seeds: &[0], // Ignored, using async buffer
            expected_number_of_new_tokens: 1,
            active_suffix_length: 1,
            sampling_start: 0,
            sampling_length: 1,
            is_prefilling: false,
        };

        let async_positions = Some((async_positions_buffer.clone(), pass_idx));
        let async_seeds = Some((async_seeds_buffer.clone(), pass_idx));

        let should_fill_attention_bias = false; // we fill it once in prepare
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
            &task.token_positions,
            task.token_bitmask,
            &task.token_seeds,
            task.active_suffix_length,
            /*sampling_start=*/ 0,
            /*sampling_length=*/ task.active_suffix_length,
            task.is_prefilling,
            None,
            skip_token_ids_copy,
            should_fill_attention_bias,
            async_positions,
            async_seeds,
        );
        if let Some(sm) = state.sampling_method_mut() {
            *sm = Some(sampling_method);
        }

        let mut command_buffer = self
            .context
            .context
            .create_command_buffer()
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?
            .start_encoding();

        // Wait on previous pass if this is a continuation
        if is_continuation {
            command_buffer.encode_wait_for_event(&self.context.async_buffers.event, current_counter);
        }

        self.context
            .executables
            .encode(&mut state, &EncodingParameters::new(), &mut command_buffer)
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;

        // Encode sampling
        self.context
            .gpu_sampler
            .encode(&mut state, &EncodingParameters::new(), &mut command_buffer)
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;

        // Copy sampled token: sampling_output → token_ids (for next pass)
        // and sampling_output → results[slot] (for callback)
        let sampling_output = state.sampling_output().expect("sampling_output must exist after sampling encode");
        let sampling_output_binding = sampling_output.borrow();
        let sampling_output_buf_rc = sampling_output_binding.buffer();
        let sampling_output_buf_borrow = sampling_output_buf_rc.borrow();
        let token_ids_binding = self.context.scratch_buffers.token_ids.borrow();
        let token_ids_buf_rc = token_ids_binding.buffer();
        let mut token_ids_buf_borrow = token_ids_buf_rc.borrow_mut();

        self.context.token_copy_sampled.encode(
            sampling_output_buf_borrow.deref(),
            token_ids_buf_borrow.deref_mut(),
            &mut command_buffer,
        );
        let results_offset = slot * std::mem::size_of::<u32>();
        self.context.token_copy_results.encode(
            sampling_output_buf_borrow.deref(),
            (results_buffer.borrow_mut().deref_mut(), results_offset),
            &mut command_buffer,
        );

        // Scatter + register for all transformer layers
        self.context.cache_layers.borrow_mut().update_after_acceptance(
            &[0],
            None,
            &mut command_buffer,
            &self.context.kv_cache_update,
        );
        self.context.cache_layers.borrow_mut().register_accepted_tokens(&[token_position]);

        if let Some(mask_update) = &self.context.mask_update {
            let updates: Vec<AttentionBiasUpdate> =
                self.context.cache_layers.borrow().attention_bias_updates_after_acceptance(1);
            for (window_size, mask_buffer) in &self.context.scratch_buffers.attention_window_size_to_bias {
                if let Some(update) = updates.iter().find(|u| &u.key == window_size) {
                    if update.unmask_col >= 0 || update.mask_col >= 0 {
                        let mask_buf_rc = mask_buffer.borrow().buffer();
                        let mut mask_buf_borrow = mask_buf_rc.borrow_mut();
                        mask_update.encode(
                            mask_buf_borrow.deref_mut(),
                            update.unmask_col,
                            update.mask_col,
                            &mut command_buffer,
                        );
                    }
                }
            }
        }

        // Signal event for next pass
        let next_counter = current_counter + 1;
        command_buffer.encode_signal_event(&self.context.async_buffers.event, next_counter);
        self.context.async_buffers.counter.set(next_counter);

        // Add completion handler
        let results_buffer_ptr = results_buffer.borrow().cpu_ptr().as_ptr() as *const u32;

        let handler = move |result: Result<&<B::CommandBuffer as CommandBuffer>::Completed, B::Error>| {
            result.expect("async decoding forward pass completed with error");
            let token = { unsafe { *results_buffer_ptr.add(slot) as u64 } };
            on_complete(token);
        };

        command_buffer.add_completion_handler(handler);
        drop(token_ids_binding);

        let pending = command_buffer.end_encoding().submit();

        if should_capture {
            pending.wait_until_completed().map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;
            self.gpu_capture.stop_capture(&self.context.context, "decode").map_err(|_| Error::CaptureFailed)?;
        }

        Ok(())
    }

    fn clear_cache(&mut self) {
        objc2::rc::autoreleasepool(|_pool| {
            self.pre_encoded_task = None;
        });
    }

    fn reset_state(&mut self) {
        self.context.cache_layers.borrow_mut().clear();
        self.tokens.clear();
        self.registered_prefix_len = 0;
        self.pre_encoded_task = None;
        self.gpu_capture.reset();

        let seed = self.decoding_config.sampling_seed.resolve();
        self.context.seed = PRng::new(seed);
        self.context.async_buffers.reset_counter();
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
        self.context.cache_layers.borrow_mut().apply_slice(slice, Some(range));
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
                    if copy_rows > 0 {
                        {
                            let mut dst_keys = dst.keys.borrow_mut();
                            let src_keys = src.keys.borrow();
                            dst_keys.copy_slice(&src_keys, 1, 0..copy_rows, 0);
                        }
                        {
                            let mut dst_values = dst.values.borrow_mut();
                            let src_values = src.values.borrow();
                            dst_values.copy_slice(&src_values, 1, 0..copy_rows, 0);
                        }
                    }
                    dst.state = src.state.clone();
                    dst.prefix_token_positions = src.prefix_token_positions.clone();
                },
                (CacheLayer::StateSpace(src), CacheLayer::StateSpace(dst)) => {
                    {
                        let mut dst_conv = dst.conv_state.borrow_mut();
                        let src_conv = src.conv_state.borrow();
                        dst_conv.copy_from_array(&src_conv);
                    }
                    {
                        let mut dst_ssm = dst.ssm_state.borrow_mut();
                        let src_ssm = src.ssm_state.borrow();
                        dst_ssm.copy_from_array(&src_ssm);
                    }
                },
                _ => panic!("Layer type mismatch when reconfiguring language model generator cache"),
            }
        }
        drop(llm_state);

        self.tokens = ctx.tokens.clone();
    }
}

impl<B: Backend> LanguageModelGenerator<B> {
    pub fn new(
        model_path: &Path,
        decoding_config: DecodingConfig,
    ) -> Result<Self, Error> {
        let gpu_capture = GpuCaptureManager::new();

        let context = LanguageModelGeneratorContext::new(model_path, &decoding_config)?;
        let prefill_step_size = decoding_config.prefill_step_size.resolve(&context.model_config);
        let generate_suffix_length = decoding_config.generate_suffix_length();

        let mut generator = Self {
            decoding_config,
            tokens: Vec::new(),
            context,
            pre_encoded_task: None,
            registered_prefix_len: 0,
            gpu_capture,
        };

        //Warmup
        generator.warmup(prefill_step_size)?;
        generator.warmup(generate_suffix_length)?;

        Ok(generator)
    }

    fn warmup(
        &mut self,
        suffix_length: usize,
    ) -> Result<(), Error> {
        let token_ids: Vec<u64> = vec![0; suffix_length];
        let token_positions: Vec<usize> = (0..suffix_length).collect();
        let token_seeds: Vec<u64> = vec![0; suffix_length];

        let task = Task {
            token_ids: &token_ids,
            token_positions: &token_positions,
            token_bitmask: None,
            token_seeds: &token_seeds,
            expected_number_of_new_tokens: suffix_length,
            active_suffix_length: suffix_length,
            sampling_start: 0,
            sampling_length: suffix_length,
            is_prefilling: false,
        };

        self.run_model(task, true, false, SamplingMethod::default(), false)?;
        Ok(())
    }

    fn run_model(
        &mut self,
        task: Task,
        warmup: bool,
        allow_pre_encode: bool,
        sampling_method: SamplingMethod,
        should_fill_attention_bias: bool,
    ) -> Result<(ForwardPassState<B>, f64), Error> {
        objc2::rc::autoreleasepool(|_pool| {
            let run_start = Instant::now();

            let mut state = ForwardPassState::new_llm(
                self.context.context.clone(),
                &self.context.decoder_config,
                &self.context.model_shape,
                &self.context.scratch_buffers,
                self.context.cache_layers.clone(),
                self.context.shared_buffers.clone(),
                task.token_ids,
                task.token_positions,
                task.token_bitmask,
                task.token_seeds,
                task.active_suffix_length,
                task.sampling_start,
                task.sampling_length,
                task.is_prefilling,
                None,
                false,
                should_fill_attention_bias,
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
                active_suffix_len: task.active_suffix_length,
                sampling_method,
                sampling_start: task.sampling_start,
                sampling_len: task.sampling_length,
                has_bitmask: task.token_bitmask.is_some(),
                is_prefilling: task.is_prefilling,
            };

            let is_first_decode = !warmup && task.token_ids.len() == 1;
            let should_capture = self.gpu_capture.should_capture_decode(is_first_decode);

            if should_capture {
                self.gpu_capture.start_capture(&self.context.context, "decode").map_err(|_| Error::CaptureFailed)?;
                self.pre_encoded_task = None;
            }

            let sample = !warmup && !task.is_prefilling;

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

            let run_time = run_start.elapsed().as_secs_f64();

            if should_capture {
                self.gpu_capture.stop_capture(&self.context.context, "decode").map_err(|_| Error::CaptureFailed)?;
            }

            Ok((state, run_time))
        })
    }

    fn encode_forward_pass(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters,
        sample: bool,
    ) -> Result<<B::CommandBuffer as CommandBuffer>::Executable, Error> {
        let mut command_buffer = self
            .context
            .context
            .create_command_buffer()
            .map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?
            .start_encoding();

        self.context
            .executables
            .encode(state, parameters, &mut command_buffer)
            .map_err(|e| Error::EncodeFailed(Box::new(e)))?;

        if sample {
            self.context
                .gpu_sampler
                .encode(state, parameters, &mut command_buffer)
                .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        }

        let executable = command_buffer.end_encoding();

        Ok(executable)
    }

    fn read_sampling_output(
        &mut self,
        state: &mut ForwardPassState<B>,
    ) -> Result<Vec<u64>, Error> {
        let sampling_output = state
            .sampling_output()
            .expect("Sampling output buffer not found - ensure sampling was encoded during forward pass");

        let output_buffer = sampling_output.borrow();
        let output_view = output_buffer.as_view::<u32>();
        let batch_size = state.sampling_length();

        let mut result = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            result.push(output_view[[i]] as u64);
        }

        Ok(result)
    }

    fn update_cache_layers(
        &mut self,
        accepted_token_indices: &[usize],
        suffix_start: Option<usize>,
        wait_until_completed: bool,
    ) -> Result<(), Error> {
        let mut command_buffer =
            self.context.context.create_command_buffer().expect("Failed to create command buffer").start_encoding();

        {
            let mut cache_layers = self.context.cache_layers.borrow_mut();
            cache_layers.update_after_acceptance(
                accepted_token_indices,
                suffix_start,
                &mut command_buffer,
                &self.context.kv_cache_update,
            );
        }

        let pending = command_buffer.end_encoding().submit();

        if wait_until_completed {
            pending.wait_until_completed().map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;
        }
        Ok(())
    }

    fn allow_pre_encode(&self) -> bool {
        let debug_active = self.context.context.debug_active();

        let result = self.decoding_config.allow_pre_encode && !debug_active;

        result
    }

    fn sync_prefix(&mut self) {
        if self.tokens.is_empty() {
            return;
        }

        let desired_prefix_len = self.tokens.len() - 1;
        if desired_prefix_len > self.registered_prefix_len {
            let positions: Vec<usize> = (self.registered_prefix_len..desired_prefix_len).collect();
            if !positions.is_empty() {
                self.context.cache_layers.borrow_mut().register_accepted_tokens(&positions);
            }
            self.registered_prefix_len = desired_prefix_len;
        }
    }

    fn should_fill_attention_bias(&self) -> bool {
        let sliding_window_sizes = self.context.model_shape.sliding_window_length_per_layer.clone();
        let has_sliding_window = sliding_window_sizes.iter().any(|size| size.is_some());
        let has_speculative_suffix = self.decoding_config.generate_suffix_length() > 1;

        has_sliding_window || has_speculative_suffix
    }

    /// Single GPU pass: updates KV cache for `prefix` and returns top-k logits for `query`.
    ///
    /// Replaces two-pass `prefill(prefix)` + `get_multi_logits(query)`, saving one round-trip.
    /// After the pass: KV is updated for `prefix` only, `self.tokens` is extended with `prefix`.
    ///
    /// Caller must ensure `prefix.len() + query.len() <= self.generate_suffix_length()`.
    pub fn extend_kv_and_get_logits(
        &mut self,
        prefix: &[u64],
        query: &[u64],
        top_k: usize,
    ) -> Vec<HashMap<u64, f32>> {
        debug_assert!(!prefix.is_empty(), "prefix must be non-empty; use get_multi_logits for empty prefix");
        debug_assert!(!query.is_empty(), "query must be non-empty");

        let cache_len = self.tokens.len();
        let total_len = prefix.len() + query.len();

        let mut token_ids: Vec<u64> = Vec::with_capacity(total_len);
        token_ids.extend_from_slice(prefix);
        token_ids.extend_from_slice(query);

        let token_positions: Vec<usize> = (cache_len..cache_len + total_len).collect();
        let token_seeds: Vec<u64> = vec![0u64; total_len];

        let task = Task {
            token_ids: &token_ids,
            token_positions: &token_positions,
            token_bitmask: None,
            token_seeds: &token_seeds,
            expected_number_of_new_tokens: total_len,
            active_suffix_length: total_len,
            sampling_start: 0,
            sampling_length: total_len,
            is_prefilling: false,
        };

        // Causal mask required whenever processing more than 1 token.
        let fill_bias = total_len > 1 || self.should_fill_attention_bias();

        let (state, _) = self.run_model(task, false, false, SamplingMethod::default(), fill_bias).unwrap();

        // Update KV cache for prefix positions only (not query — those are speculative).
        let prefix_indices: Vec<usize> = (0..prefix.len()).collect();
        self.update_cache_layers(&prefix_indices, None, true).unwrap();

        // Register prefix positions as accepted in cache layer metadata.
        let prefix_positions: Vec<usize> = (cache_len..cache_len + prefix.len()).collect();
        self.context.cache_layers.borrow_mut().register_accepted_tokens(&prefix_positions);
        self.registered_prefix_len = cache_len + prefix.len();

        // Extend self.tokens with prefix (mirrors what prefill does for KV-cache-only steps).
        self.tokens.extend_from_slice(prefix);

        self.read_logits_rows(&state, prefix.len(), query.len(), top_k)
    }

    /// Forward pass over `extra_tokens` without modifying `self.tokens` or the KV cache.
    /// Returns top-`top_k` probabilities per position.
    pub fn get_multi_logits(
        &mut self,
        extra_tokens: &[u64],
        top_k: usize,
    ) -> Vec<HashMap<u64, f32>> {
        let n = extra_tokens.len();
        if n == 0 {
            return Vec::new();
        }

        let start_pos = self.tokens.len();
        let token_positions: Vec<usize> = (start_pos..start_pos + n).collect();
        let token_seeds: Vec<u64> = vec![0u64; n];

        let task = Task {
            token_ids: extra_tokens,
            token_positions: &token_positions,
            token_bitmask: None,
            token_seeds: &token_seeds,
            expected_number_of_new_tokens: n,
            active_suffix_length: n,
            sampling_start: 0,
            // sampling_length must be > 0 for the readout kernel to populate the logits buffer.
            // We use n so that all positions get logits computed; we read them directly and
            // ignore the GPU sampling output.
            sampling_length: n,
            is_prefilling: false,
        };

        // Always fill attention bias for multi-token decode (causal mask required).
        let fill_bias = n > 1 || self.should_fill_attention_bias();

        let (state, _) = self.run_model(task, false, false, SamplingMethod::default(), fill_bias).unwrap();

        self.read_logits_rows(&state, 0, n, top_k)
    }

    /// Read `n` rows of logits from the GPU buffer starting at `row_offset`,
    /// returning the top-`top_k` token probabilities for each position.
    fn read_logits_rows(
        &self,
        state: &ForwardPassState<B>,
        row_offset: usize,
        n: usize,
        top_k: usize,
    ) -> Vec<HashMap<u64, f32>> {
        let vocab_size = self.context.model_shape.logits_shape(1)[1];
        let logits_borrow = state.llm_state().logits.borrow();
        let data_type = logits_borrow.data_type();
        let elem_size = data_type.size_in_bytes();
        let raw_bytes = logits_borrow.as_bytes();

        let mut result = Vec::with_capacity(n);
        for pos in 0..n {
            let offset = (row_offset + pos) * vocab_size * elem_size;
            let row_bytes = &raw_bytes[offset..offset + vocab_size * elem_size];

            let f32_logits: Vec<f32> = match data_type {
                crate::DataType::F32 => bytemuck::cast_slice::<u8, f32>(row_bytes).to_vec(),
                crate::DataType::F16 => {
                    bytemuck::cast_slice::<u8, half::f16>(row_bytes).iter().map(|&x| half::f16::to_f32(x)).collect()
                },
                crate::DataType::BF16 => {
                    bytemuck::cast_slice::<u8, half::bf16>(row_bytes).iter().map(|&x| half::bf16::to_f32(x)).collect()
                },
                dt => panic!("Unsupported logits data type: {dt:?}"),
            };

            result.push(logits_to_top_k_probs(f32_logits, top_k));
        }

        result
    }
}

/// Convert a full-vocabulary logit vector into a top-k softmax probability map.
///
/// `speculator_sample` expects `HashMap<u64, f32>` where values are probabilities
/// (it applies `ln(v)` before adding Gumbel noise, matching the GPU kernel which
/// applies Gumbel noise to raw logits, i.e. `logit + g = ln(softmax) + const + g`).
fn logits_to_top_k_probs(
    f32_logits: Vec<f32>,
    top_k: usize,
) -> HashMap<u64, f32> {
    let vocab_size = f32_logits.len();

    // Numerically stable softmax: subtract max before exp.
    let max_logit = f32_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = f32_logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let log_sum_exp = max_logit + sum_exp.ln();

    // Partial selection to find top-k by logit value in O(V) average time.
    let effective_k = top_k.min(vocab_size);
    let mut indexed: Vec<(u64, f32)> = f32_logits.into_iter().enumerate().map(|(i, v)| (i as u64, v)).collect();

    if effective_k < indexed.len() {
        let kth = indexed.len() - effective_k;
        indexed.select_nth_unstable_by(kth, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed[kth..].iter().map(|&(tok, logit)| (tok, (logit - log_sum_exp).exp())).collect()
    } else {
        indexed.into_iter().map(|(tok, logit)| (tok, (logit - log_sum_exp).exp())).collect()
    }
}
