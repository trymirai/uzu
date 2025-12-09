use std::{collections::HashMap, path::Path, time::Instant};

use mpsgraph::CommandBuffer;

use super::{
    LLMContext,
    gpu_capture::GpuCaptureManager,
    result::{GenerateResult, PrefillResult},
    tasks::{LLMEncodedTask, LLMRunTask},
};
use crate::{
    Array,
    backends::metal::forward_pass::{
        EncodableBlock, EncodingParameters, ForwardPassState, INVALID_POSITION,
    },
    linearizer::trie::{TokenTrie, TrieCreationConfig},
    session::{
        config::DecodingConfig,
        parameter::{ConfigResolvableValue, SamplingMethod},
        types::Error,
    },
    utils::env_utils::MetalEnvVar,
};

pub struct LLM {
    pub decoding_config: DecodingConfig,
    pub tokens: Vec<u64>,

    pub context: LLMContext,
    encoded_tasks: HashMap<String, LLMEncodedTask>,
    registered_prefix_len: usize,
    gpu_capture: GpuCaptureManager,
}

impl LLM {
    pub fn new(
        model_path: &Path,
        decoding_config: DecodingConfig,
    ) -> Result<Self, Error> {
        let gpu_capture = GpuCaptureManager::new();

        let context = LLMContext::new(model_path, &decoding_config)?;
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

            let task = LLMRunTask {
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

        let task = LLMRunTask {
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
        let task = LLMRunTask {
            token_ids: vec![0; suffix_length],
            token_positions: (0..suffix_length).collect::<Vec<usize>>(),
            token_seeds: vec![0; suffix_length],
            expected_number_of_new_tokens: suffix_length,
            active_suffix_length: suffix_length,
            is_prefilling: false,
        };

        let (_, _) =
            self.run_model(task, true, false, SamplingMethod::default());
    }

    fn run_model(
        &mut self,
        task: LLMRunTask,
        warmup: bool,
        allow_pre_encode: bool,
        sampling_method: SamplingMethod,
    ) -> (ForwardPassState, f64) {
        objc2::rc::autoreleasepool(|_pool| {
            let run_start = Instant::now();

            let mut state = task.create_state(&mut self.context, None);
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
        let sampling_output = state.sampling_output()
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
