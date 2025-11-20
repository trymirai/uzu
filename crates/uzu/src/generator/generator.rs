use std::{
    collections::HashMap, iter::repeat_n, path::Path, sync::Arc, time::Instant,
};

use block::ConcreteBlock;
use itertools::izip;
use mpsgraph::CommandBuffer;

use super::{
    context::GeneratorContext,
    gpu_capture::GpuCaptureManager,
    result::{GenerateResult, PrefillResult},
    tasks::{GeneratorEncodedTask, GeneratorRunTask},
};
use crate::{
    Array,
    backends::metal::forward_pass::{
        ForwardPassState, INVALID_POSITION,
        encodable_with_state::{EncodableWithState, EncodingParameters},
    },
    session::{
        config::DecodingConfig,
        parameter::{ConfigResolvableValue, SamplingMethod},
        types::Error,
    },
    trie::{TrieCreationConfig, TrieNode},
    utils::env_utils::MetalEnvVar,
};

pub struct Generator {
    pub decoding_config: DecodingConfig,
    pub tokens: Vec<u64>,

    pub context: GeneratorContext,
    encoded_tasks: HashMap<String, GeneratorEncodedTask>,
    registered_prefix_len: usize,
    gpu_capture: GpuCaptureManager,
}

impl Generator {
    pub fn new(
        model_path: &Path,
        decoding_config: DecodingConfig,
    ) -> Result<Self, Error> {
        let gpu_capture = GpuCaptureManager::new();

        let context = GeneratorContext::new(model_path, &decoding_config)?;
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
            &mut self.context.next_seed,
            speculator.as_ref(),
            &TrieCreationConfig::default(),
            suffix_length + 1,
        );
        let flat_trie = suffix_root.linearize();
        let active_suffix_length = flat_trie.len() - 1;

        let token_ids = tokens
            .iter()
            .copied()
            .take(tokens_length - 1)
            .chain(flat_trie.token_ids())
            .chain(repeat_n(0, suffix_length - active_suffix_length))
            .collect::<Box<[u64]>>();

        let token_positions = (prefix_offset
            ..prefix_offset + tokens_length - 1)
            .chain(flat_trie.token_positions().map(|trie_position| {
                prefix_offset + tokens_length - 1 + trie_position
            }))
            .chain(repeat_n(
                INVALID_POSITION,
                suffix_length - active_suffix_length,
            ))
            .collect::<Box<[usize]>>();

        let token_seeds = repeat_n(0, tokens_length - 1)
            .chain(flat_trie.token_seeds())
            .chain(repeat_n(0, suffix_length - active_suffix_length))
            .collect::<Box<[u64]>>();

        let mut last_state: Option<ForwardPassState> = None;
        let mut run_times: Vec<f64> = Vec::new();

        // Process each prefill step and update the KV cache.
        for (step, (step_token_ids, step_token_positions, step_token_seeds)) in
            izip!(
                token_ids.chunks(prefill_step_size),
                token_positions.chunks(prefill_step_size),
                token_seeds.chunks(prefill_step_size)
            )
            .enumerate()
        {
            let tokens_start_index = step * prefill_step_size;
            let tokens_end_index = tokens_start_index + prefill_step_size;
            let active_suffix_length = step_token_positions
                .iter()
                .position(|&pos| pos == INVALID_POSITION)
                .unwrap_or(prefill_step_size);
            let is_last_prefill_step = step == prefill_steps - 1;
            let should_sample_after_step =
                sample_suffix && is_last_prefill_step;

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

            let task = GeneratorRunTask {
                token_ids: step_token_ids,
                token_positions: step_token_positions,
                token_seeds: step_token_seeds,
                expected_number_of_new_tokens: prefill_step_size,
                active_suffix_length,
                is_prefilling: !should_sample_after_step,
            };

            let (state, run_time) = self.run_model(
                task,
                false,
                self.allow_pre_encode(),
                sampling_method,
            );

            if should_capture {
                self.gpu_capture.stop_capture("prefill");
            }

            // Register the *accepted* real tokens from this step (exclude the
            // padding token at the very end of the overall prefill).
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
            flat_trie.accept(&sampled_tokens[suffix_root_index..]);

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
        sampling_method: SamplingMethod,
    ) -> Result<GenerateResult, Error> {
        let speculator = &self.decoding_config.speculator_config.speculator;

        let suffix_length = self.decoding_config.generate_suffix_length();
        let suffix_root = TrieNode::from_speculator(
            &self.tokens,
            &mut self.context.next_seed,
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

        let task = GeneratorRunTask {
            token_ids: &token_ids,
            token_positions: &token_positions,
            token_seeds: &token_seeds,
            expected_number_of_new_tokens: 1,
            active_suffix_length,
            is_prefilling: false,
        };

        let (mut state, run_time) = self.run_model(
            task,
            false,
            self.allow_pre_encode(),
            sampling_method,
        );

        let sampled_tokens = self.sample(&mut state)?;

        let (accepted_tokens, accepted_token_indices) =
            flat_trie.accept(&sampled_tokens);

        self.update_cache_layers(&accepted_token_indices[1..], None, false);

        self.tokens.extend(accepted_tokens.clone());
        self.sync_prefix();

        Ok(GenerateResult {
            tokens: accepted_tokens,
            forwardpass_duration: run_time,
        })
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
    }

    pub fn prefix_len(&self) -> usize {
        self.registered_prefix_len
    }

    fn warmup(
        &mut self,
        suffix_length: usize,
    ) {
        let task = GeneratorRunTask {
            token_ids: &repeat_n(0, suffix_length).collect::<Box<[u64]>>(),
            token_positions: &(0..suffix_length).collect::<Box<[usize]>>(),
            token_seeds: &repeat_n(0, suffix_length).collect::<Box<[u64]>>(),
            expected_number_of_new_tokens: suffix_length,
            active_suffix_length: suffix_length,
            is_prefilling: false,
        };

        let (_, _) =
            self.run_model(task, true, false, SamplingMethod::default());
    }

    fn run_model(
        &mut self,
        task: GeneratorRunTask,
        warmup: bool,
        allow_pre_encode: bool,
        sampling_method: SamplingMethod,
    ) -> (ForwardPassState, f64) {
        objc2::rc::autoreleasepool(|_pool| {
            let run_start = Instant::now();

            let mut state = task.create_state(&mut self.context, None);
            state.sampling_method = Some(sampling_method);

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

            let root_command_buffer =
                self.context.command_buffer.root_command_buffer().to_owned();

            if !warmup {
                if !task.is_prefilling {
                    self.context.gpu_sampler.encode(
                        &mut state,
                        &self.context.command_buffer,
                        &EncodingParameters::new(warmup, true, false),
                    );
                }
            }

            self.context.command_buffer.commit_and_continue();

            if allow_pre_encode {
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
        let sampling_output = state.sampling_output.as_ref()
            .expect("Sampling output buffer not found - ensure sampling was encoded during forward pass");

        let output_buffer = sampling_output.borrow();
        let output_view = output_buffer
            .as_view::<u32>()
            .map_err(|_| Error::SamplingFailed)?;
        let batch_size = output_buffer.shape()[0];

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
        let command_buffer = CommandBuffer::from_command_queue(
            &self.context.mtl_context.command_queue,
        );
        let root_command_buffer =
            command_buffer.root_command_buffer().to_owned();

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
            command_buffer.root_command_buffer().wait_until_completed();
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

    /// Prepares async buffers for generation.
    /// Must be called after prefill, before async_generate loop.
    pub fn prepare_async(
        &mut self,
        tokens_to_generate: usize,
    ) {
        let prefill_count = self.tokens.len();
        self.context
            .async_buffers
            .prepare_positions(prefill_count, tokens_to_generate);
        self.context
            .async_buffers
            .prepare_seeds(&mut self.context.next_seed, tokens_to_generate);
        self.context.async_buffers.reset_counter();
    }

    /// Submits a single async forward pass.
    /// Does NOT block. Callback fires when GPU completes.
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

        let task = GeneratorRunTask {
            token_ids: &[last_token],
            token_positions: &[0], // Ignored, using async buffer
            token_seeds: &[0],     // Ignored, using async buffer
            expected_number_of_new_tokens: 1,
            active_suffix_length: 1,
            is_prefilling: false,
        };

        let async_positions = Some((&async_positions_buffer, pass_idx));
        let async_seeds = Some((&async_seeds_buffer, pass_idx));

        let mut state = ForwardPassState::new(
            self.context.mtl_context.clone(),
            &self.context.model_config.decoder_config,
            &self.context.model_shape,
            &self.context.scratch_buffers,
            self.context.cache_layers.clone(),
            self.context.shared_buffers.clone(),
            &task.token_ids,
            &task.token_positions,
            &task.token_seeds,
            task.active_suffix_length,
            task.is_prefilling,
            false,
            None,
            is_continuation,
            async_positions,
            async_seeds,
        );
        state.sampling_method = Some(sampling_method);

        self.context.reset_command_buffer();
        let root_command_buffer =
            self.context.command_buffer.root_command_buffer();

        // Wait on previous pass if this is a continuation
        if is_continuation {
            root_command_buffer
                .encode_wait_for_event(&async_event, current_counter);
        }

        // Encode forward pass
        self.context.executables.encode(
            &mut state,
            &self.context.command_buffer,
            &EncodingParameters::new(false, false, false),
        );

        // Encode sampling
        self.context.gpu_sampler.encode(
            &mut state,
            &self.context.command_buffer,
            &EncodingParameters::new(false, false, false),
        );

        // Copy sampled token: sampling_output → token_ids (for next pass)
        // and sampling_output → results[slot] (for callback)
        let sampling_output = state
            .sampling_output
            .as_ref()
            .expect("sampling_output must exist after sampling encode");
        let sampling_output_buffer =
            unsafe { sampling_output.borrow_mut().mtl_buffer().clone() };
        let token_ids_buffer = self.context.scratch_buffers.token_ids.clone();

        let encoder = root_command_buffer.new_compute_command_encoder();
        self.context.token_copy.encode_to_token_ids(
            &sampling_output_buffer,
            &token_ids_buffer,
            encoder,
        );
        self.context.token_copy.encode_to_results(
            &sampling_output_buffer,
            &results_buffer,
            slot,
            encoder,
        );
        encoder.end_encoding();

        // Signal event for next pass
        let next_counter = current_counter + 1;
        root_command_buffer.encode_signal_event(&async_event, next_counter);
        self.context.async_buffers.counter.set(next_counter);

        // Add completion handler
        let results_buffer_clone = results_buffer.clone();
        let callback = Arc::new(std::sync::Mutex::new(Some(on_complete)));

        let block = ConcreteBlock::new(move |_: &metal::CommandBufferRef| {
            let token = {
                let ptr = results_buffer_clone.contents() as *const u32;
                unsafe { *ptr.add(slot) as u64 }
            };
            if let Some(cb) = callback.lock().unwrap().take() {
                cb(token);
            }
        });
        let block = block.copy();

        root_command_buffer.add_completed_handler(&block);
        self.context.command_buffer.commit_and_continue();

        Ok(())
    }

    pub fn has_attention_layers(&self) -> bool {
        self.context.model_config.decoder_config.has_attention_layers()
    }
}
