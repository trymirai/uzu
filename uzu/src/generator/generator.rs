use std::{collections::HashMap, path::Path, time::Instant};

use super::{
    config::GeneratorConfig,
    context::GeneratorContext,
    result::{GenerateResult, PrefillResult},
    tasks::{GeneratorEncodedTask, GeneratorRunTask},
};
use crate::{
    Array,
    backends::metal::{
        ForwardPassState,
        forward_pass::encodable_with_state::{
            EncodableWithState, EncodingParameters,
        },
    },
    generator::error::GeneratorError,
    linearizer::trie::TokenTrie,
};

pub struct Generator {
    pub config: GeneratorConfig,
    pub tokens: Vec<u64>,

    context: GeneratorContext,
    encoded_tasks: HashMap<String, GeneratorEncodedTask>,
}

impl Generator {
    pub fn new(
        model_path: &Path,
        config: GeneratorConfig,
    ) -> Result<Self, GeneratorError> {
        let context = GeneratorContext::new(model_path, &config)?;

        let prefill_step_size = config.prefill_step_size;
        let generate_suffix_length = config.generate_suffix_length();

        let mut generator = Self {
            config: config,
            tokens: Vec::new(),
            context,
            encoded_tasks: HashMap::new(),
        };

        //Warmup
        generator.warmup(prefill_step_size);
        generator.warmup(generate_suffix_length);

        return Ok(generator);
    }

    pub fn prefill(
        &mut self,
        tokens: Vec<u64>,
        sampling_config: crate::session::sampling_config::SamplingConfig,
    ) -> PrefillResult {
        assert!(!tokens.is_empty());
        self.tokens = tokens.clone();

        let tokens_length = tokens.len();
        let number_of_prefill_steps = (tokens_length as f32
            / self.config.prefill_step_size as f32)
            .ceil() as usize;
        let total_prefill_tokens_count =
            number_of_prefill_steps * self.config.prefill_step_size;
        let unused_tokens_count = total_prefill_tokens_count - tokens_length;

        let speculator = &self.config.speculator_config.speculator;
        let proposals = speculator.generate_proposals(&tokens);
        let speculated_suffix = TokenTrie::from_sequences(&proposals)
            .linearize(0, unused_tokens_count);

        let zero_padding_tokens: Vec<u64> =
            vec![0; unused_tokens_count - speculated_suffix.tokens.len()];

        let mut padded_tokens = Vec::with_capacity(total_prefill_tokens_count);
        padded_tokens.extend_from_slice(&tokens);
        padded_tokens.extend_from_slice(&speculated_suffix.tokens);
        padded_tokens.extend_from_slice(&zero_padding_tokens);

        let mut padded_indicies: Vec<usize> = (0..tokens_length).collect();
        padded_indicies.extend(
            speculated_suffix.indices.iter().map(|index| index + tokens_length),
        );
        let zero_padding_indicies: Vec<usize> =
            vec![0; unused_tokens_count - speculated_suffix.tokens.len()];
        padded_indicies.extend(zero_padding_indicies);

        let padding_start_index = tokens.len() + speculated_suffix.tokens.len();

        let mut last_state: Option<ForwardPassState> = None;
        let mut run_times: Vec<f64> = Vec::new();

        // Process each prefill step and update KV cache immediately
        for step in 0..number_of_prefill_steps {
            let tokens_start_index = step * self.config.prefill_step_size;
            let tokens_end_index =
                tokens_start_index + self.config.prefill_step_size;
            let tokens_for_step =
                &padded_tokens[tokens_start_index..tokens_end_index];
            let indices_for_step =
                &padded_indicies[tokens_start_index..tokens_end_index];

            objc2::rc::autoreleasepool(|_pool| {
                // Drop previous state to release any transient Metal objects
                let _ = last_state.take();
            });

            let padding_mask = if step == number_of_prefill_steps - 1
                && !speculated_suffix.tokens.is_empty()
            {
                let padding_mask_vec: Vec<bool> = (0..tokens_for_step.len())
                    .map(|i| (tokens_start_index + i) >= padding_start_index)
                    .collect();
                if padding_mask_vec.iter().any(|&b| b) {
                    Some(padding_mask_vec)
                } else {
                    None
                }
            } else {
                None
            };

            let task = GeneratorRunTask {
                token_ids: tokens_for_step.to_vec(),
                token_positions: indices_for_step.to_vec(),
                padding_mask: padding_mask,
                expected_amount_of_new_tokens: self.config.prefill_step_size,
            };

            let (state, run_time) = self.run_model(
                task,
                false,
                self.allow_pre_encode(),
                Some(sampling_config.clone()),
            );

            // Register tokens with KV cache immediately after each step
            let step_end_token_index =
                std::cmp::min(tokens_end_index, tokens_length);
            let tokens_processed_this_step =
                step_end_token_index - tokens_start_index;

            if tokens_processed_this_step > 0 {
                let positions_for_step: Vec<usize> =
                    (tokens_start_index..step_end_token_index).collect();
                self.context
                    .kv_cache
                    .borrow_mut()
                    .register_accepted_tokens(&positions_for_step);
            }

            last_state = Some(state);
            run_times.push(run_time);
        }

        let mut final_state = last_state.unwrap();

        let argmax_tokens = self.gpu_sample(&mut final_state);

        let mut accepted_token_indices: Vec<usize> = Vec::new();
        let mut accepted_tokens: Vec<u64> = Vec::new();
        let mut current_token_index: isize = -1;
        loop {
            let current_index_in_window = ((current_token_index
                + tokens_length as isize)
                % self.config.prefill_step_size as isize)
                as usize;

            let new_token = argmax_tokens[current_index_in_window];
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

        self.update_kv_cache(
            &mut final_state,
            &accepted_token_indices,
            tokens_length,
        );

        // Register the final accepted tokens (from speculation)
        if !accepted_tokens.is_empty() {
            let start_pos = self.tokens.len();
            let accepted_positions: Vec<usize> =
                (0..accepted_tokens.len()).map(|i| start_pos + i).collect();
            self.context
                .kv_cache
                .borrow_mut()
                .register_accepted_tokens(&accepted_positions);
            self.tokens.extend(accepted_tokens.clone());
        }

        PrefillResult {
            tokens: accepted_tokens,
            forwardpass_durations: run_times,
        }
    }

    pub fn generate(
        &mut self,
        sampling_config: crate::session::sampling_config::SamplingConfig,
    ) -> GenerateResult {
        let last_token = self.tokens.last().unwrap();

        let speculator = &self.config.speculator_config.speculator;
        let mut proposals: Vec<Vec<u64>> = speculator
            .generate_proposals(&self.tokens)
            .into_iter()
            .map(|v| std::iter::once(*last_token).chain(v).collect())
            .collect();
        if proposals.is_empty() {
            proposals = vec![vec![*last_token]];
        }

        let speculated_suffix = TokenTrie::from_sequences(&proposals)
            .linearize(0, self.config.generate_suffix_length());

        let expected_suffix_length = self.config.generate_suffix_length();
        let unused_tokens_count =
            expected_suffix_length - speculated_suffix.tokens.len();
        let zero_padding_tokens: Vec<u64> = vec![0; unused_tokens_count];
        let padded_tokens =
            [speculated_suffix.tokens.clone(), zero_padding_tokens].concat();
        let start_position = self.tokens.len() - 1;
        let padded_indicies: Vec<usize> =
            (start_position..start_position + expected_suffix_length).collect();

        let task = GeneratorRunTask {
            token_ids: padded_tokens,
            token_positions: padded_indicies,
            padding_mask: None,
            expected_amount_of_new_tokens: 1,
        };

        let (mut state, run_time) = self.run_model(
            task,
            false,
            self.allow_pre_encode(),
            Some(sampling_config),
        );

        let argmax_tokens = self.gpu_sample(&mut state);

        let mut accepted_token_indices = Vec::new();
        let mut accepted_tokens = Vec::new();
        let mut current_token_index: isize = 0;
        loop {
            let new_token = argmax_tokens[current_token_index as usize];
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

        self.update_kv_cache(
            &mut state,
            &accepted_token_indices,
            self.tokens.len() - 1,
        );

        let start_pos = self.tokens.len();
        let accepted_positions: Vec<usize> =
            (0..accepted_tokens.len()).map(|i| start_pos + i).collect();
        self.context
            .kv_cache
            .borrow_mut()
            .register_accepted_tokens(&accepted_positions);

        self.tokens.extend(accepted_tokens.clone());

        GenerateResult {
            tokens: accepted_tokens,
            forwardpass_duration: run_time,
        }
    }

    pub fn clear_cache(&mut self) {
        objc2::rc::autoreleasepool(|_pool| {
            self.encoded_tasks.clear();
        });
    }

    fn warmup(
        &mut self,
        suffix_length: usize,
    ) {
        let task = GeneratorRunTask {
            token_ids: vec![0; suffix_length],
            token_positions: (0..suffix_length).collect::<Vec<usize>>(),
            padding_mask: None,
            expected_amount_of_new_tokens: suffix_length,
        };

        let (_, _) = self.run_model(task, true, false, None);
    }

    fn run_model(
        &mut self,
        task: GeneratorRunTask,
        warmup: bool,
        _allow_pre_encode: bool,
        sampling_config: Option<
            crate::session::sampling_config::SamplingConfig,
        >,
    ) -> (ForwardPassState, f64) {
        objc2::rc::autoreleasepool(|_pool| {
            let run_start = Instant::now();

            let mut state = task.create_state(&mut self.context);
            state.sampling_config = sampling_config;

            self.context.reset_command_buffer();
            task.build_encoded_task(
                &self.context,
                &mut state,
                &EncodingParameters::new(warmup, true, false),
            );

            let root_command_buffer =
                self.context.command_buffer.root_command_buffer().to_owned();

            if !warmup {
                self.context.gpu_sampler.encode(
                    &mut state,
                    &self.context.command_buffer,
                    &EncodingParameters::new(warmup, true, false),
                );
            }

            self.context.command_buffer.commit_and_continue();

            root_command_buffer.wait_until_completed();

            let run_time = run_start.elapsed().as_secs_f64();
            (state, run_time)
        })
    }

    fn gpu_sample(
        &mut self,
        state: &mut ForwardPassState,
    ) -> Vec<u64> {
        let sampling_output = state.sampling_output.as_ref()
            .expect("Sampling output buffer not found - ensure sampling was encoded during forward pass");

        let output_buffer = sampling_output.borrow();
        let output_view = output_buffer.as_view::<u32>().unwrap();
        let batch_size = output_buffer.shape()[0];

        let mut result = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            result.push(output_view[[i]] as u64);
        }

        result
    }

    fn update_kv_cache(
        &mut self,
        _state: &mut ForwardPassState,
        accepted_token_indices: &[usize],
        tokens_count: usize,
    ) {
        let root_command_buffer =
            self.context.command_buffer.root_command_buffer().to_owned();

        self.context.kv_cache.borrow_mut().update_after_acceptance(
            accepted_token_indices,
            &self.context.command_buffer,
            &self.context.kv_cache_update,
            tokens_count,
        );

        self.context.command_buffer.commit_and_continue();
        root_command_buffer.wait_until_completed();
    }

    fn allow_pre_encode(&self) -> bool {
        // let metal_debug_active =
        //     env_utils::MetalEnvVar::DeviceWrapperType.is_enabled();

        // let result = self.config.allow_pre_encode
        //     && self.context.kernels_config.use_attention
        //     && !metal_debug_active;

        // result

        false
    }
}
