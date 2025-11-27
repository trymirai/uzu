use std::{collections::HashMap, iter::repeat_n, path::Path, time::Instant};

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
    generator::grammar::CompiledGrammar,
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
            &mut self.context.next_seed,
            compiled_grammar.as_deref_mut(),
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
            .collect::<Vec<u64>>();

        let token_bitmask = if compiled_grammar.is_some() {
            Some(
                repeat_n(
                    vec![0; self.context.model_shape.bitmask_shape(1)[1]],
                    tokens_length - 1,
                )
                .chain(flat_trie.token_masks().unwrap().map(|x| x.to_vec()))
                .chain(repeat_n(
                    vec![0; self.context.model_shape.bitmask_shape(1)[1]],
                    suffix_length - active_suffix_length,
                ))
                .collect::<Vec<Vec<u32>>>()
                .concat(),
            )
        } else {
            None
        };

        let token_positions = (prefix_offset
            ..prefix_offset + tokens_length - 1)
            .chain(flat_trie.token_positions().map(|trie_position| {
                prefix_offset + tokens_length - 1 + trie_position
            }))
            .chain(repeat_n(
                INVALID_POSITION,
                suffix_length - active_suffix_length,
            ))
            .collect::<Vec<usize>>();

        let token_seeds = repeat_n(0, tokens_length - 1)
            .chain(flat_trie.token_seeds())
            .chain(repeat_n(0, suffix_length - active_suffix_length))
            .collect::<Vec<u64>>();

        let mut last_state: Option<ForwardPassState> = None;
        let mut run_times: Vec<f64> = Vec::new();

        // Process each prefill step and update the KV cache.
        for step in 0..prefill_steps {
            let tokens_start_index = step * prefill_step_size;
            let tokens_end_index = tokens_start_index + prefill_step_size;
            let tokens_for_step =
                &token_ids[tokens_start_index..tokens_end_index];
            let positions_for_step =
                &token_positions[tokens_start_index..tokens_end_index];
            let active_suffix_length = positions_for_step
                .iter()
                .position(|&pos| pos == INVALID_POSITION)
                .unwrap_or(prefill_step_size);
            let bitmask_for_step = token_bitmask.as_ref().map(|x| {
                x[tokens_start_index
                    * self.context.model_shape.bitmask_shape(1)[1]
                    ..tokens_end_index
                        * self.context.model_shape.bitmask_shape(1)[1]]
                    .to_vec()
            });
            let seeds_for_step =
                &token_seeds[tokens_start_index..tokens_end_index];
            let is_last_prefill_step = step == prefill_steps - 1;
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
                token_bitmask: bitmask_for_step,
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

        let mut accepted_token_indices = vec![suffix_root_index];
        let mut accepted_tokens = vec![sampled_tokens[suffix_root_index]];
        if let Some(compiled_grammar) = compiled_grammar.as_deref_mut() {
            compiled_grammar.accept_token(sampled_tokens[suffix_root_index]);
        }
        let mut current_token = &suffix_root;

        while let Some(next_token) =
            current_token.get(*accepted_tokens.last().unwrap())
        {
            let next_token_index =
                suffix_root_index + flat_trie.index(next_token).unwrap();

            accepted_tokens.push(sampled_tokens[next_token_index]);
            accepted_token_indices.push(next_token_index);
            if let Some(compiled_grammar) = compiled_grammar.as_deref_mut() {
                compiled_grammar.accept_token(sampled_tokens[next_token_index]);
            }

            current_token = next_token;
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
        mut compiled_grammar: Option<&mut CompiledGrammar>,
        sampling_method: SamplingMethod,
    ) -> Result<GenerateResult, Error> {
        let speculator = &self.decoding_config.speculator_config.speculator;

        let suffix_length = self.decoding_config.generate_suffix_length();
        let suffix_root = TrieNode::from_speculator(
            &self.tokens,
            &mut self.context.next_seed,
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
            .collect();

        let token_bitmask = if compiled_grammar.is_some() {
            Some(
                flat_trie
                    .token_masks()
                    .unwrap()
                    .map(|x| x.to_vec())
                    .chain(repeat_n(
                        vec![0; self.context.model_shape.bitmask_shape(1)[1]],
                        suffix_length - active_suffix_length,
                    ))
                    .collect::<Vec<Vec<u32>>>()
                    .concat(),
            )
        } else {
            None
        };

        let start_position = self.tokens.len() - 1;
        let token_positions = flat_trie
            .token_positions()
            .map(|trie_position| start_position + trie_position)
            .chain(repeat_n(
                INVALID_POSITION,
                suffix_length - active_suffix_length,
            ))
            .collect();

        let token_seeds = flat_trie
            .token_seeds()
            .chain(repeat_n(0, suffix_length - active_suffix_length))
            .collect();

        let task = GeneratorRunTask {
            token_ids,
            token_positions,
            token_bitmask,
            token_seeds,
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
        let mut current_token = &suffix_root;
        loop {
            let current_token_index = flat_trie.index(current_token).unwrap();
            let current_token_id = sampled_tokens[current_token_index];

            accepted_token_indices.push(current_token_index);
            accepted_tokens.push(current_token_id);
            if let Some(compiled_grammar) = compiled_grammar.as_deref_mut() {
                compiled_grammar.accept_token(current_token_id);
            }

            let Some(next_token) = current_token.get(current_token_id) else {
                break;
            };

            current_token = next_token;
        }

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
            token_ids: vec![0; suffix_length],
            token_positions: (0..suffix_length).collect::<Vec<usize>>(),
            token_bitmask: None,
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
}
