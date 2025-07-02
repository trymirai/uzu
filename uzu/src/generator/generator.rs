use std::{collections::HashMap, path::Path, time::Instant};
use half::f16;
use crate::env_utils::MetalEnvVar;
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
use mpsgraph::CommandBuffer;

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

        self.update_kv_cache(&mut final_state, &accepted_token_indices);

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

        self.update_kv_cache(&mut state, &accepted_token_indices);

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
        allow_pre_encode: bool,
        sampling_config: Option<
            crate::session::sampling_config::SamplingConfig,
        >,
    ) -> (ForwardPassState, f64) {
        objc2::rc::autoreleasepool(|_pool| {
            let run_start = Instant::now();

            let mut state = task.create_state(&mut self.context, None);
            state.sampling_config = sampling_config;

            let encoded_task_key = task.encoded_task_key(self.tokens.len());
            
            if let Some(_) = self.encoded_tasks.remove(&encoded_task_key) {
                //Nothing
            } else {
                eprintln!("building encoded task for key: {}", encoded_task_key);
                self.context.reset_command_buffer();

                _ = task.build_encoded_task(
                    &self.context,
                    &mut state,
                    &EncodingParameters::new(warmup, true, false),
                    encoded_task_key.clone()
                );
            }

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

            if allow_pre_encode {
                let next_task_key: String = task.encoded_task_key(self.tokens.len() + 1);

                eprintln!("=== PRE-ENCODING NEXT TASK ===");
                eprintln!("Next task key: {}", next_task_key);
                eprintln!("Using projection step: 1");
                
                // The projection_step parameter should handle the simulation without modifying KV cache
                eprintln!("Pre-encoding with projection_step=1, current prefix: {}", 
                    self.context.kv_cache.borrow().data[0].effective_prefix_length());

                let next_encoded_task = task.build_encoded_task(
                    &self.context,
                    &mut state,
                    &EncodingParameters::new(warmup, false, false).with_projection(1),
                    next_task_key.clone()
                );

                self.encoded_tasks.insert(next_task_key.clone(), next_encoded_task);
                eprintln!("=== PRE-ENCODING COMPLETE ===");
            }

            root_command_buffer.wait_until_completed();
            let run_time = run_start.elapsed().as_secs_f64();
            
            // Debug prints after wait_until_completed
            eprintln!("=== DEBUG AFTER WAIT_UNTIL_COMPLETED ===");
            eprintln!("Current tokens length: {}", self.tokens.len());
            
            // Print prefix lengths for first few layers
            for layer_idx in 0..std::cmp::min(3, self.context.kv_cache.borrow().data.len()) {
                let layer = &self.context.kv_cache.borrow().data[layer_idx];
                eprintln!("Layer {}: effective_prefix_length = {}, projected(0) = {}, projected(1) = {}", 
                    layer_idx, 
                    layer.effective_prefix_length(),
                    layer.projected_effective_prefix_length(0),
                    layer.projected_effective_prefix_length(1)
                );
            }
            
            // Print attention bias mask values (around prefix boundary)
            let attention_bias_binding = state.hashmaps(&[crate::backends::metal::forward_pass::HashMapId::AttentionBias]);
            let attention_bias_map = &attention_bias_binding[0];
            
            // Get the actual prefix length for the first layer
            let actual_prefix_len = self.context.kv_cache.borrow().data[0].effective_prefix_length();
            let suffix_len = state.arrays(&[crate::backends::metal::forward_pass::ArrayId::QKV])[0].borrow().shape()[0];
            let total_seq_len = actual_prefix_len + suffix_len;
            
            for (window_key, bias_array) in attention_bias_map.iter() {
                let bias_borrowed = bias_array.borrow();
                let shape = bias_borrowed.shape();
                eprintln!("Attention bias window {:?}: shape {:?}, actual_prefix_len: {}, suffix_len: {}, total_seq_len: {}", 
                    window_key, shape, actual_prefix_len, suffix_len, total_seq_len);
                
                if shape.len() >= 2 && total_seq_len <= shape[1] {
                    // Show values around the prefix boundary (prefix_len-4 to prefix_len+4)
                    let start_col = actual_prefix_len.saturating_sub(4);
                    let end_col = std::cmp::min(actual_prefix_len + 4, total_seq_len);
                    print!("First row around prefix boundary [{}..{}]: [", start_col, end_col);
                    
                    // Try f16 first, then f32
                    if let Ok(bias_view) = bias_borrowed.as_view::<f16>() {
                        for col in start_col..end_col {
                            print!("{:.1}, ", bias_view[[0, col]].to_f32());
                        }
                    } else if let Ok(bias_view) = bias_borrowed.as_view::<f32>() {
                        for col in start_col..end_col {
                            print!("{:.1}, ", bias_view[[0, col]]);
                        }
                    } else {
                        print!("unknown type");
                    }
                    println!("]");
                }
                break; // Just print first bias matrix
            }
            
            // Print KV cache values (around actual prefix positions)
            if !self.context.kv_cache.borrow().data.is_empty() {
                let first_layer = &self.context.kv_cache.borrow().data[0];
                let keys_borrowed = first_layer.keys.borrow();
                let values_borrowed = first_layer.values.borrow();
                let keys_shape = keys_borrowed.shape();
                let values_shape = values_borrowed.shape();
                
                eprintln!("KV cache keys shape: {:?}", keys_shape);
                eprintln!("KV cache values shape: {:?}", values_shape);
                
                // Show values around the actual prefix length
                if keys_shape.len() >= 3 && actual_prefix_len > 0 && keys_shape[2] >= 1 {
                    let start_seq = actual_prefix_len.saturating_sub(4);
                    let end_seq = std::cmp::min(actual_prefix_len + 4, keys_shape[1]);
                    print!("Keys[0] around prefix [{}..{}], first dim: [", start_seq, end_seq);
                    
                    // Try f16 first, then f32
                    if let Ok(keys_view) = keys_borrowed.as_view::<f16>() {
                        for seq in start_seq..end_seq {
                            print!("{:.3}, ", keys_view[[0, seq, 0]].to_f32());
                        }
                    } else if let Ok(keys_view) = keys_borrowed.as_view::<f32>() {
                        for seq in start_seq..end_seq {
                            print!("{:.3}, ", keys_view[[0, seq, 0]]);
                        }
                    } else {
                        print!("unknown type");
                    }
                    println!("]");
                }
                
                if values_shape.len() >= 3 && actual_prefix_len > 0 && values_shape[2] >= 1 {
                    let start_seq = actual_prefix_len.saturating_sub(4);
                    let end_seq = std::cmp::min(actual_prefix_len + 4, values_shape[1]);
                    print!("Values[0] around prefix [{}..{}], first dim: [", start_seq, end_seq);
                    
                    // Try f16 first, then f32
                    if let Ok(values_view) = values_borrowed.as_view::<f16>() {
                        for seq in start_seq..end_seq {
                            print!("{:.3}, ", values_view[[0, seq, 0]].to_f32());
                        }
                    } else if let Ok(values_view) = values_borrowed.as_view::<f32>() {
                        for seq in start_seq..end_seq {
                            print!("{:.3}, ", values_view[[0, seq, 0]]);
                        }
                    } else {
                        print!("unknown type");
                    }
                    println!("]");
                }
            }
            eprintln!("=== END DEBUG ===");
            
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
    ) {
        let command_buffer = CommandBuffer::from_command_queue(&self.context.mtl_context.command_queue);

        self.context.kv_cache.borrow_mut().update_after_acceptance(
            accepted_token_indices,
            &command_buffer,
            &self.context.kv_cache_update,
        );

        command_buffer.commit_and_continue();
    }

    fn allow_pre_encode(&self) -> bool {
        let metal_debug_active =
            MetalEnvVar::DeviceWrapperType.is_enabled();

        let result = self.config.allow_pre_encode
            && self.context.kernels_config.use_attention
            && !metal_debug_active;

        result
    }
}
