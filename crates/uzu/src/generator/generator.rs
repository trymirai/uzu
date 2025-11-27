use std::{collections::HashMap, path::Path, sync::Arc, time::Instant};

use block::ConcreteBlock;
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
    linearizer::trie::{TokenTrie, TrieCreationConfig},
    session::{
        config::DecodingConfig,
        parameter::{ConfigResolvableValue, SamplingMethod},
        types::Error,
    },
    utils::env_utils::MetalEnvVar,
};

pub struct Generator {
    pub decoding_config: DecodingConfig,
    pub tokens: Vec<u64>,

    pub context: GeneratorContext,
    encoded_tasks: HashMap<String, GeneratorEncodedTask>,
    registered_prefix_len: usize,
    gpu_capture: GpuCaptureManager,

    async_prefill_count: usize,
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
            async_prefill_count: 0,
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

        let prefill_step_size = self
            .decoding_config
            .prefill_step_size
            .resolve(&self.context.model_config);

        let tokens_length = tokens.len();
        let number_of_prefill_steps =
            (tokens_length as f32 / prefill_step_size as f32).ceil() as usize;
        let total_prefill_tokens_count =
            number_of_prefill_steps * prefill_step_size;
        let unused_tokens_count = total_prefill_tokens_count - tokens_length;

        let speculator = &self.decoding_config.speculator_config.speculator;
        let speculated_suffix = TokenTrie::from_speculator(
            &tokens,
            &mut self.context.next_seed,
            false,
            speculator.as_ref(),
            &TrieCreationConfig::default(),
            unused_tokens_count,
        )
        .linearize(0, unused_tokens_count);

        let zero_padding_tokens: Vec<u64> =
            vec![0; unused_tokens_count - speculated_suffix.tokens.len()];

        let padded_tokens = [
            &tokens[..],
            &speculated_suffix.tokens[..],
            &zero_padding_tokens[..],
        ]
        .concat();

        let mut padded_positions: Vec<usize> =
            (prefix_offset..prefix_offset + tokens_length).collect();
        padded_positions.extend(
            speculated_suffix
                .indices
                .iter()
                .map(|index| index + prefix_offset + tokens_length),
        );
        let padding_count =
            unused_tokens_count - speculated_suffix.tokens.len();
        padded_positions
            .extend(std::iter::repeat(INVALID_POSITION).take(padding_count));

        let zero_padding_left_seeds: Vec<u64> = vec![0; tokens_length];
        let zero_padding_right_seeds: Vec<u64> =
            vec![0; unused_tokens_count - speculated_suffix.seeds.len()];
        let padded_seeds = [
            &zero_padding_left_seeds[..],
            &speculated_suffix.seeds[..],
            &zero_padding_right_seeds[..],
        ]
        .concat();

        let mut last_state: Option<ForwardPassState> = None;
        let mut run_times: Vec<f64> = Vec::new();

        // Process each prefill step and update the KV cache.
        for step in 0..number_of_prefill_steps {
            let tokens_start_index = step * prefill_step_size;
            let tokens_end_index = tokens_start_index + prefill_step_size;
            let tokens_for_step =
                &padded_tokens[tokens_start_index..tokens_end_index];
            let positions_for_step =
                &padded_positions[tokens_start_index..tokens_end_index];
            let active_suffix_length = positions_for_step
                .iter()
                .position(|&pos| pos == INVALID_POSITION)
                .unwrap_or(prefill_step_size);
            let seeds_for_step =
                &padded_seeds[tokens_start_index..tokens_end_index];
            let is_last_prefill_step = step == number_of_prefill_steps - 1;
            let should_sample_after_step =
                sample_suffix && is_last_prefill_step;

            let is_first_prefill = step == 0;
            let should_capture =
                self.gpu_capture.should_capture_prefill(is_first_prefill);

            if should_capture {
                let _ = self
                    .gpu_capture
                    .start_capture(&self.context.mtl_context, "prefill");
            }

            objc2::rc::autoreleasepool(|_pool| {
                let _ = last_state.take();
            });

            let task = GeneratorRunTask {
                token_ids: tokens_for_step.to_vec(),
                token_positions: positions_for_step.to_vec(),
                token_seeds: seeds_for_step.to_vec(),
                expected_number_of_new_tokens: prefill_step_size,
                active_suffix_length,
                is_prefilling: !should_sample_after_step,
            };

            let (state, run_time) = self.run_model(
                task,
                false,
                self.allow_pre_encode(),
                is_last_prefill_step,
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
                if step == number_of_prefill_steps - 1 && sample_suffix {
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

        let last_suffix_start =
            prefill_step_size * (number_of_prefill_steps - 1);

        let sampling_row = (tokens_length - last_suffix_start) - 1;
        let mut current_token_index: isize = -1;
        let mut accepted_token_indices: Vec<usize> = vec![sampling_row];
        let mut accepted_tokens: Vec<u64> = Vec::new();
        loop {
            let current_index_in_window =
                ((current_token_index + tokens_length as isize)
                    % prefill_step_size as isize) as usize;
            let new_token = sampled_tokens[current_index_in_window];
            accepted_tokens.push(new_token);

            if let Some(map) =
                speculated_suffix.transition_map.get(&current_token_index)
            {
                if let Some(&next_token_index) = map.get(&new_token) {
                    accepted_token_indices.push(next_token_index as usize);
                    current_token_index = next_token_index;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        self.update_cache_layers(
            &accepted_token_indices,
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
        let speculated_suffix = TokenTrie::from_speculator(
            &self.tokens,
            &mut self.context.next_seed,
            true,
            speculator.as_ref(),
            &TrieCreationConfig::default(),
            self.decoding_config.generate_suffix_length(),
        )
        .linearize(0, self.decoding_config.generate_suffix_length());

        let expected_suffix_length =
            self.decoding_config.generate_suffix_length();
        let unused_tokens_count =
            expected_suffix_length - speculated_suffix.tokens.len();

        let zero_padding_tokens: Vec<u64> = vec![0; unused_tokens_count];
        let padded_tokens =
            [speculated_suffix.tokens.clone(), zero_padding_tokens].concat();

        let zero_padding_seeds: Vec<u64> = vec![0; unused_tokens_count];
        let padded_seeds =
            [speculated_suffix.seeds.clone(), zero_padding_seeds].concat();

        let start_position = self.tokens.len() - 1;

        let padded_positions: Vec<usize> = speculated_suffix
            .indices
            .iter()
            .map(|idx| idx + start_position)
            .chain(
                std::iter::repeat(INVALID_POSITION).take(unused_tokens_count),
            )
            .collect();
        let active_suffix_length = speculated_suffix.tokens.len();

        let task = GeneratorRunTask {
            token_ids: padded_tokens,
            token_positions: padded_positions,
            token_seeds: padded_seeds,
            expected_number_of_new_tokens: 1,
            active_suffix_length,
            is_prefilling: false,
        };

        let (mut state, run_time) = self.run_model(
            task,
            false,
            self.allow_pre_encode(),
            true,
            sampling_method,
        );

        let sampled_tokens = self.sample(&mut state)?;

        let mut accepted_token_indices = Vec::new();
        let mut accepted_tokens = Vec::new();
        let mut current_token_index: isize = 0;
        loop {
            let new_token = sampled_tokens[current_token_index as usize];
            accepted_tokens.push(new_token);
            if let Some(map) =
                speculated_suffix.transition_map.get(&(current_token_index))
            {
                if let Some(&next_token_index) = map.get(&new_token) {
                    accepted_token_indices.push(next_token_index as usize);
                    current_token_index = next_token_index;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        self.update_cache_layers(&accepted_token_indices, None, false);

        self.tokens.extend(accepted_tokens.clone());
        self.sync_prefix();

        Ok(GenerateResult {
            tokens: accepted_tokens,
            forwardpass_duration: run_time,
        })
    }

    pub fn async_generate<F>(
        &mut self,
        pass_idx: usize,
        sampling_method: SamplingMethod,
        on_complete: F,
    ) -> Result<(), Error>
    where
        F: FnOnce(u64) + Send + 'static,
    {
        assert!(
            self.decoding_config.generate_suffix_length() == 1,
            "async_generate only supports single token generation (suffix_length=1)"
        );

        let current_counter = self.context.async_event_counter.get();
        let is_continuation = current_counter > 0;

        let last_token = *self.tokens.last().ok_or(Error::PrefillFailed)?;

        let task = GeneratorRunTask {
            token_ids: vec![last_token],
            token_positions: vec![0], // Not used - comes from async_positions_buffer
            token_seeds: vec![0], // Not used - comes from async_seeds_buffer
            expected_number_of_new_tokens: 1,
            active_suffix_length: 1,
            is_prefilling: false,
        };

        let mut state = task.create_state_async(
            &mut self.context,
            pass_idx,
            is_continuation,
        );
        state.sampling_method = Some(sampling_method);

        self.context.reset_command_buffer();

        let root_cmd = self.context.command_buffer.root_command_buffer();

        if is_continuation {
            root_cmd.encode_wait_for_event(
                &self.context.async_event,
                current_counter,
            );
        }

        _ = task.build_encoded_task(
            &self.context,
            &mut state,
            &EncodingParameters::new(false, true, false),
            task.encoded_task_key(self.tokens.len()),
        );

        self.context.gpu_sampler.encode(
            &mut state,
            &self.context.command_buffer,
            &EncodingParameters::new(false, true, false),
        );

        let sampling_output_buffer = {
            let sampling_output = state
                .sampling_output
                .as_ref()
                .expect("Sampling output buffer not found");
            let mut buffer = sampling_output.borrow_mut();
            unsafe { buffer.mtl_buffer().to_owned() }
        };

        let token_ids_buffer = self.context.scratch_buffers.token_ids.clone();
        let results_buffer = self.context.async_results_buffer.clone();

        {
            let root_cmd = self.context.command_buffer.root_command_buffer();
            let encoder = root_cmd.new_compute_command_encoder();
            self.context.token_copy.encode(
                &sampling_output_buffer,
                &token_ids_buffer,
                encoder,
            );
            self.context.token_copy.encode_to_offset(
                &sampling_output_buffer,
                &results_buffer,
                pass_idx,
                encoder,
            );
            encoder.end_encoding();
        }

        let next_counter = current_counter + 1;
        root_cmd.encode_signal_event(&self.context.async_event, next_counter);
        self.context.async_event_counter.set(next_counter);

        let callback = Arc::new(std::sync::Mutex::new(Some(on_complete)));
        let callback_clone = callback.clone();

        let block =
            ConcreteBlock::new(move |_cmd_buf: &metal::CommandBufferRef| {
                let token = unsafe {
                    let ptr = results_buffer.contents() as *const u32;
                    let token_ptr = ptr.add(pass_idx);
                    *token_ptr as u64
                };
                if let Some(cb) = callback_clone.lock().unwrap().take() {
                    cb(token);
                }
            });
        let block = block.copy();

        let root_command_buffer =
            self.context.command_buffer.root_command_buffer();
        root_command_buffer.add_completed_handler(&block);

        self.context.command_buffer.commit_and_continue();

        Ok(())
    }

    pub fn update_last_token(
        &mut self,
        token: u64,
    ) {
        if let Some(last) = self.tokens.last_mut() {
            *last = token;
        }
    }

    pub fn set_async_prefill_count(&mut self) {
        self.async_prefill_count = self.tokens.len();
    }

    pub fn async_generate_batch<F>(
        &mut self,
        count: usize,
        sampling_method: SamplingMethod,
        on_complete: F,
    ) -> Result<(), Error>
    where
        F: Fn(usize, u64) + Send + Sync + 'static,
    {
        assert!(
            self.decoding_config.generate_suffix_length() == 1,
            "async_generate_batch only supports single token generation"
        );
        assert!(count > 0, "count must be > 0");

        let sampling_output_buffer =
            self.context.scratch_buffers.sampling_output.clone();
        let token_ids_buffer = self.context.scratch_buffers.token_ids.clone();
        let callback = Arc::new(on_complete);

        let start_position = self.tokens.len() - 1;

        for i in 0..count {
            self.context.reset_command_buffer();

            let position = start_position + i;
            let seed = self.context.next_seed.next();

            if i == 0 {
                let last_token =
                    *self.tokens.last().ok_or(Error::PrefillFailed)?;
                let task = GeneratorRunTask {
                    token_ids: vec![last_token],
                    token_positions: vec![position],
                    token_seeds: vec![seed],
                    expected_number_of_new_tokens: 1,
                    active_suffix_length: 1,
                    is_prefilling: false,
                };

                let mut state = task.create_state(&mut self.context, None);
                state.sampling_method = Some(sampling_method);

                _ = task.build_encoded_task(
                    &self.context,
                    &mut state,
                    &EncodingParameters::new(false, true, false),
                    task.encoded_task_key(self.tokens.len() + i),
                );

                self.context.gpu_sampler.encode(
                    &mut state,
                    &self.context.command_buffer,
                    &EncodingParameters::new(false, true, false),
                );
            } else {
                let task = GeneratorRunTask {
                    token_ids: vec![0],
                    token_positions: vec![position],
                    token_seeds: vec![seed],
                    expected_number_of_new_tokens: 1,
                    active_suffix_length: 1,
                    is_prefilling: false,
                };

                let mut state = task.create_state(&mut self.context, None);
                state.sampling_method = Some(sampling_method);

                _ = task.build_encoded_task(
                    &self.context,
                    &mut state,
                    &EncodingParameters::new(false, true, false),
                    task.encoded_task_key(self.tokens.len() + i),
                );

                self.context.gpu_sampler.encode(
                    &mut state,
                    &self.context.command_buffer,
                    &EncodingParameters::new(false, true, false),
                );
            }

            {
                let root_cmd =
                    self.context.command_buffer.root_command_buffer();
                let encoder = root_cmd.new_compute_command_encoder();
                self.context.token_copy.encode(
                    &sampling_output_buffer,
                    &token_ids_buffer,
                    encoder,
                );
                encoder.end_encoding();
            }

            let callback_clone = callback.clone();
            let buf_clone = sampling_output_buffer.clone();
            let idx = i;

            let block = ConcreteBlock::new(
                move |_cmd_buf: &metal::CommandBufferRef| {
                    let token = unsafe {
                        let ptr = buf_clone.contents() as *const u32;
                        *ptr as u64
                    };
                    callback_clone(idx, token);
                },
            );
            let block = block.copy();

            let root_cmd = self.context.command_buffer.root_command_buffer();
            root_cmd.add_completed_handler(&block);

            self.context.command_buffer.commit_and_continue();

            self.tokens.push(0);
        }

        Ok(())
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
        self.context.reset_async_event_counter();
    }

    pub fn prefix_len(&self) -> usize {
        self.registered_prefix_len
    }

    fn warmup(
        &mut self,
        suffix_length: usize,
    ) {
        let task = GeneratorRunTask {
            token_ids: vec![0; suffix_length],
            token_positions: (0..suffix_length).collect::<Vec<usize>>(),
            token_seeds: vec![0; suffix_length],
            expected_number_of_new_tokens: suffix_length,
            active_suffix_length: suffix_length,
            is_prefilling: false,
        };

        let (_, _) =
            self.run_model(task, true, false, true, SamplingMethod::default());
    }

    fn run_model(
        &mut self,
        task: GeneratorRunTask,
        warmup: bool,
        allow_pre_encode: bool,
        need_wait_until_completed: bool,
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

            if need_wait_until_completed {
                root_command_buffer.wait_until_completed();
            }
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
}
