use std::{collections::HashMap, path::Path, time::Instant};

use mpsgraph::CommandBuffer;

use super::{
    context::GeneratorContext,
    result::{GenerateResult, PrefillResult},
    tasks::{GeneratorEncodedTask, GeneratorRunTask},
};
use crate::{
    Array,
    backends::metal::forward_pass::{
        ForwardPassState,
        encodable_with_state::{EncodableWithState, EncodingParameters},
        kv_cache::INVALID_POSITION,
    },
    linearizer::trie::TokenTrie,
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
    pending_kv_update: Option<u64>,
}

impl Generator {
    pub fn new(
        model_path: &Path,
        decoding_config: DecodingConfig,
    ) -> Result<Self, Error> {
        let context = GeneratorContext::new(model_path, &decoding_config)?;

        let prefill_step_size =
            decoding_config.prefill_step_size.resolve(&context.model_config);
        let generate_suffix_length = decoding_config.generate_suffix_length();

        let mut generator = Self {
            decoding_config: decoding_config,
            tokens: Vec::new(),
            context,
            encoded_tasks: HashMap::new(),
            registered_prefix_len: 0,
            pending_kv_update: None,
        };

        //Warmup
        generator.warmup(prefill_step_size);
        generator.warmup(generate_suffix_length);

        return Ok(generator);
    }

    pub fn prefill(
        &mut self,
        tokens: Vec<u64>,
        sampling_method: SamplingMethod,
        prefix_offset: usize,
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
        let proposals = speculator.generate_proposals(&tokens);
        let speculated_suffix = TokenTrie::from_sequences(&proposals)
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

            objc2::rc::autoreleasepool(|_pool| {
                let _ = last_state.take();
            });

            let task = GeneratorRunTask {
                token_ids: tokens_for_step.to_vec(),
                token_positions: positions_for_step.to_vec(),
                expected_number_of_new_tokens: prefill_step_size,
            };

            let (mut state, run_time) = self.run_model(
                task,
                false,
                self.allow_pre_encode(),
                sampling_method,
            );

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
                if step == number_of_prefill_steps - 1 {
                    // Exclude the last token because it belongs to the suffix for sampling.
                    positions_for_step.pop();
                }

                if !positions_for_step.is_empty() {
                    self.context
                        .kv_cache
                        .borrow_mut()
                        .register_accepted_tokens(&positions_for_step);
                    let accept_indices_for_step: Vec<usize> =
                        (0..positions_for_step.len()).collect();
                    if !accept_indices_for_step.is_empty() {
                        self.update_kv_cache(
                            &mut state,
                            &accept_indices_for_step,
                        );
                    }
                    if let Some(&last_idx) = positions_for_step.last() {
                        self.registered_prefix_len = last_idx + 1;
                    }
                }
            }

            last_state = Some(state);
            run_times.push(run_time);
        }

        let mut final_state = last_state.ok_or(Error::PrefillFailed)?;
        let sampled_tokens = self.sample(&mut final_state)?;

        let mut accepted_token_indices: Vec<usize> = Vec::new();
        let mut accepted_tokens: Vec<u64> = Vec::new();
        let mut current_token_index: isize = -1;
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

        self.update_kv_cache(&mut final_state, &accepted_token_indices);

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
        let last_token = self.tokens.last().ok_or(Error::GenerateFailed)?;

        let speculator = &self.decoding_config.speculator_config.speculator;
        let mut proposals: Vec<Vec<u64>> = speculator
            .generate_proposals(&self.tokens)
            .into_iter()
            .map(|v| std::iter::once(*last_token).chain(v).collect())
            .collect();
        if proposals.is_empty() {
            proposals = vec![vec![*last_token]];
        }

        let speculated_suffix = TokenTrie::from_sequences(&proposals)
            .linearize(0, self.decoding_config.generate_suffix_length());

        let expected_suffix_length =
            self.decoding_config.generate_suffix_length();
        let unused_tokens_count =
            expected_suffix_length - speculated_suffix.tokens.len();
        let zero_padding_tokens: Vec<u64> = vec![0; unused_tokens_count];
        let padded_tokens =
            [speculated_suffix.tokens.clone(), zero_padding_tokens].concat();

        let start_position = self.tokens.len() - 1;

        let padded_positions: Vec<usize> = speculated_suffix
            .indices
            .iter()
            .map(|idx| idx + start_position)
            .chain(
                std::iter::repeat(INVALID_POSITION).take(unused_tokens_count),
            )
            .collect();

        let task = GeneratorRunTask {
            token_ids: padded_tokens,
            token_positions: padded_positions,
            expected_number_of_new_tokens: 1,
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

        self.update_kv_cache(&mut state, &accepted_token_indices);

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
        self.context.kv_cache.borrow_mut().clear();
        self.tokens.clear();
        self.registered_prefix_len = 0;
        self.encoded_tasks.clear();
        self.pending_kv_update = None;
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
            expected_number_of_new_tokens: suffix_length,
        };

        let (_, _) = self.run_model(task, true, false, SamplingMethod::Greedy);
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

            let t_create_state = Instant::now();
            let mut state = task.create_state(&mut self.context, None);
            state.sampling_method = Some(sampling_method);
            eprintln!("[RunModel] Create state: {:.3}ms", t_create_state.elapsed().as_secs_f64() * 1000.0);

            let encoded_task_key = task.encoded_task_key(self.tokens.len());

            let t_encode = Instant::now();
            if let Some(_) = self.encoded_tasks.remove(&encoded_task_key) {
                //Nothing
            } else {
                self.context.reset_command_buffer();

                _ = task.build_encoded_task(
                    &self.context,
                    &mut state,
                    &EncodingParameters::new(warmup, true, false),
                    encoded_task_key.clone(),
                );
            }
            eprintln!("[RunModel] Encode: {:.3}ms", t_encode.elapsed().as_secs_f64() * 1000.0);

            let root_command_buffer =
                self.context.command_buffer.root_command_buffer().to_owned();

            if let Some(signal_value) = self.pending_kv_update.take() {
                root_command_buffer.encode_wait_for_event(
                    &self.context.kv_update_event,
                    signal_value,
                );
            }

            if !warmup {
                self.context.gpu_sampler.encode(
                    &mut state,
                    &self.context.command_buffer,
                    &EncodingParameters::new(warmup, true, false),
                );
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

            let t_wait = Instant::now();
            root_command_buffer.wait_until_completed();
            eprintln!("[RunModel] GPU wait: {:.3}ms", t_wait.elapsed().as_secs_f64() * 1000.0);
            
            let run_time = run_start.elapsed().as_secs_f64();
            eprintln!("[RunModel] TOTAL run_model: {:.3}ms", run_time * 1000.0);

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

    fn update_kv_cache(
        &mut self,
        _state: &mut ForwardPassState,
        accepted_token_indices: &[usize],
    ) {
        let command_buffer = CommandBuffer::from_command_queue(
            &self.context.mtl_context.command_queue,
        );
        let root_command_buffer = command_buffer.command_buffer().to_owned();

        {
            let mut kv_cache = self.context.kv_cache.borrow_mut();
            kv_cache.update_after_acceptance(
                accepted_token_indices,
                &command_buffer,
                &self.context.kv_cache_update,
            );
        }

        let signal_value = self.context.kv_update_signal;
        root_command_buffer
            .encode_signal_event(&self.context.kv_update_event, signal_value);
        self.context.kv_update_signal += 1;
        self.pending_kv_update = Some(signal_value);

        command_buffer.commit();
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
                    .kv_cache
                    .borrow_mut()
                    .register_accepted_tokens(&positions);
            }
            self.registered_prefix_len = desired_prefix_len;
        }
    }
}
