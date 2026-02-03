use std::{
    collections::HashMap, iter::repeat_n, path::Path, sync::Arc, time::Instant,
};

use itertools::{Either, Itertools, izip};

use super::{
    LanguageModelGeneratorContext,
    gpu_capture::GpuCaptureManager,
    grammar::CompiledGrammar,
    result::{GenerateResult, PrefillResult},
    rng::PRng,
    tasks::{LanguageModelGeneratorEncodedTask, LanguageModelGeneratorRunTask},
};
use crate::{
    Array,
    backends::{
        common::kernel::{MaskUpdateKernel, MaskUpdateParams},
        metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandBufferExt,
            MTLCommandBufferHandler, MTLCommandEncoder, MTLCommandQueue,
            forward_pass::{
                AttentionBiasUpdate, EncodableBlock, EncodingParameters,
                ForwardPassState, INVALID_POSITION,
            },
        },
    },
    session::{
        config::DecodingConfig,
        parameter::{ConfigResolvableValue, ResolvableValue, SamplingMethod},
        types::Error,
    },
    trie::{TrieCreationConfig, TrieNode},
    utils::env_utils::MetalEnvVar,
};

pub struct LanguageModelGenerator {
    pub decoding_config: DecodingConfig,
    pub tokens: Vec<u64>,

    pub context: LanguageModelGeneratorContext,
    encoded_tasks: HashMap<String, LanguageModelGeneratorEncodedTask>,
    registered_prefix_len: usize,
    gpu_capture: GpuCaptureManager,
}

impl LanguageModelGenerator {
    pub fn new(
        model_path: &Path,
        decoding_config: DecodingConfig,
    ) -> Result<Self, Error> {
        let gpu_capture = GpuCaptureManager::new();

        let context =
            LanguageModelGeneratorContext::new(model_path, &decoding_config)?;
        let prefill_step_size =
            decoding_config.prefill_step_size.resolve(&context.model_config);
        let generate_suffix_length = decoding_config.generate_suffix_length();

        let mut generator = Self {
            decoding_config,
            tokens: Vec::new(),
            context,
            encoded_tasks: HashMap::new(),
            registered_prefix_len: 0,
            gpu_capture,
        };

        //Warmup
        generator.warmup(prefill_step_size);
        generator.warmup(generate_suffix_length);

        Ok(generator)
    }

    pub fn prefill(
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

        let prefill_step_size = self
            .decoding_config
            .prefill_step_size
            .resolve(&self.context.model_config);
        let prefill_steps = tokens_length.div_ceil(prefill_step_size);
        let prefill_size = prefill_steps * prefill_step_size;

        let speculator = &self.decoding_config.speculator_config.speculator;

        let suffix_length = prefill_size - tokens_length;
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

        let token_ids = tokens
            .iter()
            .copied()
            .take(tokens_length - 1)
            .chain(flat_trie.token_ids())
            .chunks(prefill_step_size);

        let token_positions = (prefix_offset
            ..prefix_offset + tokens_length - 1)
            .chain(flat_trie.token_positions().map(|trie_position| {
                prefix_offset + tokens_length - 1 + trie_position
            }))
            .chunks(prefill_step_size);

        let single_token_bitmask_size =
            self.context.model_shape.bitmask_shape(1)[1];
        let token_bitmasks = repeat_n(None, tokens_length - 1)
            .chain(flat_trie.token_masks())
            .chunks(prefill_step_size);

        let token_seeds = repeat_n(0, tokens_length - 1)
            .chain(flat_trie.token_seeds())
            .chunks(prefill_step_size);

        let mut last_state: Option<ForwardPassState> = None;
        let mut run_times: Vec<f64> = Vec::new();

        // Process each prefill step and update the KV cache.
        for (
            step,
            (
                step_token_ids,
                step_token_positions,
                step_token_bitmasks,
                step_token_seeds,
            ),
        ) in
            izip!(&token_ids, &token_positions, &token_bitmasks, &token_seeds)
                .enumerate()
        {
            let tokens_start_index = step * prefill_step_size;
            let tokens_end_index = tokens_start_index + prefill_step_size;

            let step_token_ids = step_token_ids.collect::<Box<[u64]>>();
            let step_token_positions =
                step_token_positions.collect::<Box<[usize]>>();
            let step_token_seeds = step_token_seeds.collect::<Box<[u64]>>();

            let active_suffix_length = step_token_positions.len();
            let is_last_prefill_step = step == prefill_steps - 1;
            let should_sample_after_step =
                sample_suffix && is_last_prefill_step;

            // If we sample on the last prefill step, we only need logits/sampling
            // for tokens that are beyond the prompt prefix (i.e. starting at the
            // suffix-root token, which is the last prompt token).
            let (sampling_start, sampling_length) = if should_sample_after_step
            {
                let suffix_root_index_in_step =
                    (tokens_length - 1).saturating_sub(tokens_start_index);
                let sampling_length = active_suffix_length
                    .saturating_sub(suffix_root_index_in_step);
                debug_assert!(
                    sampling_length > 0,
                    "Expected at least one token to sample on the last prefill step"
                );
                (suffix_root_index_in_step, sampling_length)
            } else {
                (0, 0)
            };

            let step_token_bitmask: Option<Box<[u32]>> =
                if has_grammar && sampling_length > 0 {
                    Some(
                        step_token_bitmasks
                            .map(|mask| match mask {
                                Some(mask) => Either::Left(
                                    mask.iter()
                                        .copied()
                                        .take(single_token_bitmask_size)
                                        .chain(repeat_n(
                                            0u32,
                                            single_token_bitmask_size
                                                .saturating_sub(mask.len()),
                                        )),
                                ),
                                None => Either::Right(repeat_n(
                                    u32::MAX,
                                    single_token_bitmask_size,
                                )),
                            })
                            .flatten()
                            .collect::<Box<[u32]>>(),
                    )
                } else {
                    // Drain the chunk iterator to keep the other chunked iterators aligned.
                    let _ = step_token_bitmasks.count();
                    None
                };

            let should_capture =
                self.gpu_capture.should_capture_prefill(step == 0);

            if should_capture {
                let _ = self
                    .gpu_capture
                    .start_capture(&self.context.mtl_context, "prefill");
            }

            objc2::rc::autoreleasepool(|_pool| {
                let _ = last_state.take();
            });

            let task = LanguageModelGeneratorRunTask {
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
                self.skip_attention_bias_fill(),
            );

            if should_capture {
                self.gpu_capture.stop_capture("prefill");
            }

            // Register the accepted prompt tokens from this step.
            let step_end_token_index =
                std::cmp::min(tokens_end_index, tokens_length);
            let tokens_processed_this_step =
                step_end_token_index - tokens_start_index;

            if tokens_processed_this_step > 0 {
                let mut positions_for_step: Vec<usize> = (tokens_start_index
                    ..step_end_token_index)
                    .map(|idx| idx + prefix_offset)
                    .collect();
                if step == prefill_steps - 1 && sample_suffix {
                    // Exclude the last token because it belongs to the suffix for sampling.
                    positions_for_step.pop();
                }

                if !positions_for_step.is_empty() {
                    let accept_indices_for_step: Vec<usize> =
                        (0..positions_for_step.len()).collect();
                    if !accept_indices_for_step.is_empty() {
                        self.update_cache_layers(
                            &accept_indices_for_step,
                            None,
                            true,
                        );
                    }

                    self.context
                        .cache_layers
                        .borrow_mut()
                        .register_accepted_tokens(&positions_for_step);

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
        let sampled_tokens = self.sample(&mut final_state)?;

        let last_suffix_start = prefill_step_size * (prefill_steps - 1);
        let suffix_root_index = (tokens_length - last_suffix_start) - 1;

        let (accepted_tokens, accepted_token_indices) =
            flat_trie.accept(&sampled_tokens, compiled_grammar.as_deref_mut());

        self.update_cache_layers(
            &accepted_token_indices
                .into_iter()
                .map(|p| suffix_root_index + p)
                .collect::<Box<[usize]>>(),
            Some(last_suffix_start),
            false,
        );

        self.tokens.extend(accepted_tokens.clone());
        self.sync_prefix();

        Ok(PrefillResult {
            tokens: accepted_tokens,
            forwardpass_durations: run_times,
        })
    }

    pub fn generate(
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

        let flat_trie = suffix_root.linearize();
        let active_suffix_length = flat_trie.len();

        let token_ids = flat_trie
            .token_ids()
            .chain(repeat_n(0, suffix_length - active_suffix_length))
            .collect::<Box<[u64]>>();

        let token_bitmask: Option<Box<[u32]>> =
            compiled_grammar.is_some().then(|| {
                let single_token_bitmask_size =
                    self.context.model_shape.bitmask_shape(1)[1];
                flat_trie
                    .token_masks()
                    .chain(repeat_n(None, suffix_length - active_suffix_length))
                    .map(|mask| match mask {
                        Some(mask) => Either::Left(
                            mask.iter()
                                .copied()
                                .take(single_token_bitmask_size)
                                .chain(repeat_n(
                                    0u32,
                                    single_token_bitmask_size
                                        .saturating_sub(mask.len()),
                                )),
                        ),
                        None => Either::Right(repeat_n(
                            u32::MAX,
                            single_token_bitmask_size,
                        )),
                    })
                    .flatten()
                    .collect::<Box<[u32]>>()
            });

        let start_position = self.tokens.len() - 1;
        let token_positions = flat_trie
            .token_positions()
            .map(|trie_position| start_position + trie_position)
            .chain(repeat_n(
                INVALID_POSITION,
                suffix_length - active_suffix_length,
            ))
            .collect::<Box<[usize]>>();

        let token_seeds = flat_trie
            .token_seeds()
            .chain(repeat_n(0, suffix_length - active_suffix_length))
            .collect::<Box<[u64]>>();

        let task = LanguageModelGeneratorRunTask {
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

        let (mut state, run_time) = self.run_model(
            task,
            false,
            self.allow_pre_encode(),
            sampling_method,
            self.skip_attention_bias_fill(),
        );

        let sampled_tokens = self.sample(&mut state)?;

        let (accepted_tokens, accepted_token_indices) =
            flat_trie.accept(&sampled_tokens, compiled_grammar.as_deref_mut());

        self.update_cache_layers(&accepted_token_indices, None, false);

        self.tokens.extend(accepted_tokens.clone());
        self.sync_prefix();

        Ok(GenerateResult {
            tokens: accepted_tokens,
            forwardpass_duration: run_time,
        })
    }

    /// Prepares async buffers for generation.
    /// Must be called after prefill, before async_generate loop.
    pub fn prepare_async(
        &mut self,
        tokens_to_generate: usize,
    ) {
        let prefill_count = self.tokens.len();

        // Initialize attention bias buffers to zero for async mode
        // This ensures unwritten columns (beyond what pass 0 fills) are 0 (attend)
        // rather than garbage (which could be -inf and mask incorrectly)
        for (_, mask_buffer) in
            &self.context.scratch_buffers.attention_window_size_to_bias
        {
            unsafe {
                let ptr =
                    mask_buffer.borrow().backend_buffer().contents().as_ptr()
                        as *mut u8;
                std::ptr::write_bytes(
                    ptr,
                    0,
                    mask_buffer.borrow().backend_buffer().length() as usize,
                );
            }
        }

        self.context
            .async_buffers
            .prepare_positions(prefill_count, tokens_to_generate);
        self.context.async_buffers.prepare_seeds(
            &self.context.seed,
            prefill_count,
            tokens_to_generate,
        );
        self.context.async_buffers.reset_counter();
    }

    /// Submits a single async forward pass.
    /// Does NOT block (except when GPU capture is enabled for the first decode).
    ///
    /// - `pass_idx`: Index of this pass (0, 1, 2, ...)
    /// - `sampling_method`: Sampling configuration
    /// - `on_complete`: Callback receiving sampled token as u64
    pub fn async_generate<F>(
        &mut self,
        pass_idx: usize,
        sampling_method: SamplingMethod,
        on_complete: F,
    ) -> Result<(), Error>
    where
        F: FnOnce(u64) + Send + 'static,
    {
        assert_eq!(
            self.decoding_config.generate_suffix_length(),
            1,
            "async_generate only supports suffix_length=1"
        );

        // Extract values from async_buffers before mutable borrow
        let current_counter = self.context.async_buffers.counter.get();
        let is_continuation = current_counter > 0;
        let batch_size = self.context.async_buffers.batch_size;
        let slot = pass_idx % batch_size;
        let async_event = self.context.async_buffers.event.clone();
        let results_buffer = self.context.async_buffers.results.clone();
        let async_positions_buffer =
            self.context.async_buffers.positions.clone();
        let async_seeds_buffer = self.context.async_buffers.seeds.clone();

        let last_token = *self.tokens.last().ok_or(Error::PrefillFailed)?;

        let token_position = unsafe {
            let ptr = async_positions_buffer.contents().as_ptr() as *const u32;
            *ptr.add(pass_idx) as usize
        };

        let task = LanguageModelGeneratorRunTask {
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

        let async_positions = Some((&async_positions_buffer, pass_idx));
        let async_seeds = Some((&async_seeds_buffer, pass_idx));

        let skip_attention_bias_fill =
            pass_idx > 0 && self.skip_attention_bias_fill();

        let skip_token_ids_copy = pass_idx > 0;

        let is_first_decode = !is_continuation;
        let should_capture =
            self.gpu_capture.should_capture_decode(is_first_decode);
        if should_capture {
            let _ = self
                .gpu_capture
                .start_capture(&self.context.mtl_context, "decode");
        }

        let mut state = ForwardPassState::new_llm(
            self.context.mtl_context.clone(),
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
            skip_attention_bias_fill,
            async_positions,
            async_seeds,
        );
        if let Some(sm) = state.sampling_method_mut() {
            *sm = Some(sampling_method);
        }

        self.context.reset_command_buffer();
        let root_command_buffer = self.context.command_buffer.clone();

        // Wait on previous pass if this is a continuation
        if is_continuation {
            root_command_buffer
                .encode_wait_for_event_value(&async_event, current_counter);
        }

        self.context.executables.encode(
            &mut state,
            &root_command_buffer,
            &EncodingParameters::new(false, false, false),
        );

        // Encode sampling
        self.context.gpu_sampler.encode(
            &mut state,
            &root_command_buffer,
            &EncodingParameters::new(false, false, false),
        );

        // Copy sampled token: sampling_output → token_ids (for next pass)
        // and sampling_output → results[slot] (for callback)
        let sampling_output = state
            .sampling_output()
            .expect("sampling_output must exist after sampling encode");
        let sampling_output_buffer =
            sampling_output.borrow().mtl_buffer_cloned();
        let token_ids_buffer =
            self.context.scratch_buffers.token_ids.borrow().mtl_buffer_cloned();

        let encoder = root_command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        self.context.token_copy.encode_to_token_ids(
            &sampling_output_buffer,
            &token_ids_buffer,
            &encoder,
        );
        self.context.token_copy.encode_to_results(
            &sampling_output_buffer,
            &results_buffer,
            slot,
            &encoder,
        );
        encoder.end_encoding();

        // Scatter + register for all transformer layers
        {
            let cb = root_command_buffer.clone();
            self.context.cache_layers.borrow_mut().update_after_acceptance(
                &[0],
                None,
                &cb,
                &self.context.kv_cache_update,
            );
            self.context
                .cache_layers
                .borrow_mut()
                .register_accepted_tokens(&[token_position]);
        }

        let encoder = root_command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");

        if let Some(mask_update) = &self.context.mask_update {
            let updates: Vec<AttentionBiasUpdate> = self
                .context
                .cache_layers
                .borrow()
                .attention_bias_updates_after_acceptance(1);
            for (window_size, mask_buffer) in
                &self.context.scratch_buffers.attention_window_size_to_bias
            {
                if let Some(update) =
                    updates.iter().find(|u| &u.key == window_size)
                {
                    if update.unmask_col >= 0 || update.mask_col >= 0 {
                        mask_update.encode(
                            mask_buffer.borrow().backend_buffer(),
                            &MaskUpdateParams {
                                unmask_col: update.unmask_col,
                                mask_col: update.mask_col,
                            },
                            &encoder,
                        );
                    }
                }
            }
        }

        encoder.end_encoding();

        // Signal event for next pass
        let next_counter = current_counter + 1;
        root_command_buffer
            .encode_signal_event_value(&async_event, next_counter);
        self.context.async_buffers.counter.set(next_counter);

        // Add completion handler
        let results_buffer_clone = results_buffer.clone();
        let callback = Arc::new(std::sync::Mutex::new(Some(on_complete)));

        let handler = MTLCommandBufferHandler::new(move |_| {
            let token = {
                let ptr =
                    results_buffer_clone.contents().as_ptr() as *const u32;
                unsafe { *ptr.add(slot) as u64 }
            };
            if let Some(cb) = callback.lock().unwrap().take() {
                cb(token);
            }
        });

        root_command_buffer.add_completed_handler(&handler);
        root_command_buffer.commit();

        if should_capture {
            root_command_buffer.wait_until_completed();
            self.gpu_capture.stop_capture("decode");
        }

        Ok(())
    }

    pub fn has_attention_layers(&self) -> bool {
        self.context.decoder_config.has_attention_layers()
    }

    pub fn clear_cache(&mut self) {
        objc2::rc::autoreleasepool(|_pool| {
            self.encoded_tasks.clear();
        });
    }

    pub fn reset_state(&mut self) {
        self.context.cache_layers.borrow_mut().clear();
        self.tokens.clear();
        self.registered_prefix_len = 0;
        self.encoded_tasks.clear();
        self.gpu_capture.reset();

        let seed = self.decoding_config.sampling_seed.resolve();
        self.context.seed = PRng::new(seed);
        self.context.async_buffers.reset_counter();
    }

    pub fn prefix_len(&self) -> usize {
        self.registered_prefix_len
    }

    fn warmup(
        &mut self,
        suffix_length: usize,
    ) {
        let token_ids: Vec<u64> = vec![0; suffix_length];
        let token_positions: Vec<usize> = (0..suffix_length).collect();
        let token_seeds: Vec<u64> = vec![0; suffix_length];

        let task = LanguageModelGeneratorRunTask {
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

        let (_, _) =
            self.run_model(task, true, false, SamplingMethod::default(), false);
    }

    fn run_model(
        &mut self,
        task: LanguageModelGeneratorRunTask,
        warmup: bool,
        allow_pre_encode: bool,
        sampling_method: SamplingMethod,
        skip_attention_bias_fill: bool,
    ) -> (ForwardPassState, f64) {
        objc2::rc::autoreleasepool(|_pool| {
            let run_start = Instant::now();

            let mut state = task.create_state(
                &mut self.context,
                None,
                skip_attention_bias_fill,
            );
            if let Some(method) = state.sampling_method_mut() {
                *method = Some(sampling_method);
            }

            let is_first_decode = !warmup && task.token_ids.len() == 1;
            let should_capture =
                self.gpu_capture.should_capture_decode(is_first_decode);

            if should_capture {
                let _ = self
                    .gpu_capture
                    .start_capture(&self.context.mtl_context, "decode");
            }

            let encoded_task_key = task.encoded_task_key(self.tokens.len());

            if should_capture {
                self.encoded_tasks.remove(&encoded_task_key);
            }

            if let Some(_) = self.encoded_tasks.remove(&encoded_task_key) {
                // Nothing
            } else {
                self.context.reset_command_buffer();

                _ = task.build_encoded_task(
                    &self.context,
                    &mut state,
                    &EncodingParameters::new(warmup, true, false),
                    encoded_task_key.clone(),
                );
            }

            let root_command_buffer = self.context.command_buffer.clone();

            if !warmup {
                if !task.is_prefilling {
                    self.context.gpu_sampler.encode(
                        &mut state,
                        &self.context.command_buffer,
                        &EncodingParameters::new(warmup, true, false),
                    );
                }
            }

            self.context.command_buffer.commit();

            if allow_pre_encode {
                self.context.reset_command_buffer();

                let next_task_key: String =
                    task.encoded_task_key(self.tokens.len() + 1);

                let next_encoded_task = task.build_encoded_task(
                    &self.context,
                    &mut state,
                    &EncodingParameters::new(warmup, false, false)
                        .with_projection(1),
                    next_task_key.clone(),
                );

                self.encoded_tasks
                    .insert(next_task_key.clone(), next_encoded_task);
            }

            root_command_buffer.wait_until_completed();
            let run_time = run_start.elapsed().as_secs_f64();
            if should_capture {
                self.gpu_capture.stop_capture("decode");
            }

            (state, run_time)
        })
    }

    fn sample(
        &mut self,
        state: &mut ForwardPassState,
    ) -> Result<Vec<u64>, Error> {
        let sampling_output = state.sampling_output()
            .expect("Sampling output buffer not found - ensure sampling was encoded during forward pass");

        let output_buffer = sampling_output.borrow();
        let output_view = output_buffer
            .as_view::<u32>()
            .map_err(|_| Error::SamplingFailed)?;
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
    ) {
        let command_buffer = self
            .context
            .mtl_context
            .command_queue
            .command_buffer()
            .expect("Failed to create command buffer")
            .to_owned();
        let root_command_buffer = command_buffer.to_owned();

        {
            let mut cache_layers = self.context.cache_layers.borrow_mut();
            cache_layers.update_after_acceptance(
                accepted_token_indices,
                suffix_start,
                &root_command_buffer,
                &self.context.kv_cache_update,
            );
        }

        command_buffer.commit();

        if wait_until_completed {
            command_buffer.wait_until_completed();
        }
    }

    fn allow_pre_encode(&self) -> bool {
        let metal_debug_active = MetalEnvVar::DeviceWrapperType.is_enabled();

        let result =
            self.decoding_config.allow_pre_encode && !metal_debug_active;

        result
    }

    fn sync_prefix(&mut self) {
        if self.tokens.is_empty() {
            return;
        }

        let desired_prefix_len = self.tokens.len() - 1;
        if desired_prefix_len > self.registered_prefix_len {
            let positions: Vec<usize> =
                (self.registered_prefix_len..desired_prefix_len).collect();
            if !positions.is_empty() {
                self.context
                    .cache_layers
                    .borrow_mut()
                    .register_accepted_tokens(&positions);
            }
            self.registered_prefix_len = desired_prefix_len;
        }
    }

    fn skip_attention_bias_fill(&self) -> bool {
        let sliding_window_sizes =
            self.context.model_shape.sliding_window_length_per_layer.clone();
        let has_sliding_window =
            sliding_window_sizes.iter().any(|size| size.is_some());
        let has_speculative_suffix =
            self.decoding_config.generate_suffix_length() > 1;
        let should_skip = !has_sliding_window && !has_speculative_suffix;
        return should_skip;
    }
}
