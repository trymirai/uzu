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
    env_utils,
    generator::{
        error::GeneratorError,
        mask_descriptor::{GeneratorMaskDescriptor, GeneratorPrefixLength},
    },
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
        let prefix_indices: Vec<usize> = (0..tokens_length).collect();
        let proposals = speculator.generate_proposals(&tokens);
        let speculated_suffix = TokenTrie::from_sequences(&proposals)
            .linearize(0, unused_tokens_count);
        let speculated_suffix_indicies: Vec<usize> = speculated_suffix
            .indices
            .iter()
            .map(|index| index + tokens_length)
            .collect();
        let mut indicies = prefix_indices.clone();
        indicies.extend(speculated_suffix_indicies);

        let zero_padding_tokens: Vec<u64> =
            vec![0; unused_tokens_count - speculated_suffix.tokens.len()];
        let zero_padding_indicies: Vec<usize> =
            vec![0; unused_tokens_count - speculated_suffix.tokens.len()];
        let padded_tokens =
            [tokens, speculated_suffix.tokens, zero_padding_tokens].concat();
        let padded_indicies = [indicies, zero_padding_indicies].concat();

        let mut last_state: Option<ForwardPassState> = None;
        let mut run_times: Vec<f64> = Vec::new();
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

            let casual_mask: Option<Vec<Vec<bool>>>;
            if step == number_of_prefill_steps - 1 {
                let speculated_tokens_start =
                    self.tokens.len() % self.config.prefill_step_size;
                casual_mask = Some(
                    GeneratorMaskDescriptor::prefill_last_step_casual_mask(
                        self.config.prefill_step_size,
                        speculated_tokens_start,
                        speculated_suffix.causal_mask.clone(),
                    ),
                );
            } else {
                casual_mask = None
            }

            let prefix_length = tokens_start_index;
            let mask_descriptor = GeneratorMaskDescriptor {
                suffix_length: self.config.prefill_step_size,
                prefix_length: GeneratorPrefixLength {
                    real: prefix_length,
                    step: None,
                },
                casual_mask,
            };

            let task = GeneratorRunTask {
                token_ids: tokens_for_step.to_vec(),
                token_positions: indices_for_step.to_vec(),
                mask_descriptor,
                expected_amount_of_new_tokens: self.config.prefill_step_size,
            };

            let (state, run_time) = self.run_model(
                task,
                false,
                self.allow_pre_encode(),
                Some(sampling_config.clone()),
            );

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
            self.tokens.len(),
        );
        self.update_ring_buffers(accepted_tokens.len());

        self.register_tokens(accepted_tokens.clone());

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
        let zero_padding_indicies: Vec<usize> = vec![0; unused_tokens_count];
        let padded_tokens =
            [speculated_suffix.tokens, zero_padding_tokens].concat();
        let padded_indicies: Vec<usize> =
            [speculated_suffix.indices, zero_padding_indicies]
                .concat()
                .iter()
                .map(|index| index + self.tokens.len() - 1)
                .collect();

        let prefix_length = self.tokens.len() - 1;

        let casual_mask: Option<Vec<Vec<bool>>> =
            Some(speculated_suffix.causal_mask);

        let mask_descriptor = GeneratorMaskDescriptor {
            suffix_length: expected_suffix_length,
            prefix_length: GeneratorPrefixLength {
                real: prefix_length,
                step: self.config.prefix_length_step,
            },
            casual_mask,
        };

        let task = GeneratorRunTask {
            token_ids: padded_tokens,
            token_positions: padded_indicies,
            mask_descriptor,
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
        self.update_ring_buffers(accepted_tokens.len());

        self.register_tokens(accepted_tokens.clone());

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
        let mask_descriptor = GeneratorMaskDescriptor {
            suffix_length,
            prefix_length: GeneratorPrefixLength {
                real: 0,
                step: None,
            },
            casual_mask: None,
        };

        let task = GeneratorRunTask {
            token_ids: vec![0; suffix_length],
            token_positions: (0..suffix_length).collect::<Vec<usize>>(),
            mask_descriptor,
            expected_amount_of_new_tokens: suffix_length,
        };

        let (_, _) = self.run_model(task, true, false, None);
    }

    fn run_model(
        &mut self,
        task: GeneratorRunTask,
        warmup: bool,
        allow_pre_encode: bool,
        sampling_config: Option<
            crate::session::sampling_config::SamplingConfig,
        >,
    ) -> (ForwardPassState, f64) {
        objc2::rc::autoreleasepool(|_pool| {
            let run_start = Instant::now();

            let mut state = task.create_state(&mut self.context);
            state.sampling_config = sampling_config;

            let encoded_task_key = task.encoded_task_key();
            if let Some(_) = self.encoded_tasks.remove(&encoded_task_key) {
                //Nothing
            } else {
                self.context.reset_command_buffer();

                _ = task.build_encoded_task(
                    &self.context,
                    &mut state,
                    &EncodingParameters::new(warmup, true, false),
                );
            }

            let root_command_buffer =
                self.context.command_buffer.root_command_buffer().to_owned();

            if let Some(_) = task.mask_descriptor.kv_cache_update_task() {
                self.context.kv_cache_update.encode(
                    &mut state,
                    &self.context.command_buffer,
                    &EncodingParameters::new(warmup, true, false),
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
                let next_task = task.speculate_next_task();
                next_task.apply_to_context(&mut self.context);

                let next_encoded_task = next_task.build_encoded_task(
                    &mut self.context,
                    &mut state,
                    &EncodingParameters::new(warmup, false, false),
                );

                let next_task_key = next_task.encoded_task_key();
                self.encoded_tasks.insert(next_task_key, next_encoded_task);
            }

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

    fn register_tokens(
        &mut self,
        tokens: Vec<u64>,
    ) {
        self.tokens.extend(tokens);
    }

    fn update_kv_cache(
        &mut self,
        state: &mut ForwardPassState,
        accepted_token_indices: &[usize],
        tokens_count: usize,
    ) {
        if accepted_token_indices.is_empty() {
            return;
        }

        let source_indices: Vec<usize> = accepted_token_indices
            .iter()
            .map(|index| index + tokens_count)
            .collect();
        let destination_indices: Vec<usize> = (self.tokens.len()
            ..self.tokens.len() + accepted_token_indices.len())
            .collect();
        if source_indices == destination_indices {
            return;
        }

        state.kv_cache_update_source_indices = source_indices;
        state.kv_cache_update_destination_indices = destination_indices;

        let root_command_buffer =
            self.context.command_buffer.root_command_buffer().to_owned();
        self.context.kv_cache_update.encode(
            state,
            &self.context.command_buffer,
            &EncodingParameters::new(false, true, false),
        );
        self.context.command_buffer.commit_and_continue();
        root_command_buffer.wait_until_completed();
    }

    fn allow_pre_encode(&self) -> bool {
        let metal_debug_active =
            env_utils::MetalEnvVar::DeviceWrapperType.is_enabled();

        let result = self.config.allow_pre_encode
            && self.context.kernels_config.use_attention
            && !metal_debug_active;

        result
    }

    fn update_ring_buffers(
        &mut self,
        suffix_length: usize,
    ) {
        let mut kv_cache = self.context.kv_cache.borrow_mut();
        for layer in kv_cache.data.iter_mut() {
            layer.update_after_acceptance(suffix_length);
        }
    }
}
