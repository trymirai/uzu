use std::{path::Path, rc::Rc};

use super::{
    config::GeneratorConfig,
    result::{GenerateResult, PrefillResult},
};
use crate::{
    Array,
    backends::{
        Backend, Context, RunResult, SamplingConfig, kv_cache::INVALID_POSITION,
    },
    generator::error::GeneratorError,
    linearizer::trie::TokenTrie,
    parameters::ParameterLoader,
    utils::{load_decoder_config, open_weights_file},
};

pub struct Generator<B: Backend> {
    pub config: GeneratorConfig,
    pub tokens: Vec<u64>,
    backend: B,
}

impl<B: Backend> Generator<B> {
    pub fn new(
        model_path: &Path,
        config: GeneratorConfig,
    ) -> Result<Self, GeneratorError> {
        let context = Rc::new(
            B::Context::default()
                .ok_or(GeneratorError::UnableToCreateMetalContext)?,
        );

        let decoder_config = load_decoder_config(model_path)
            .map_err(|_| GeneratorError::UnableToLoadConfig)?;

        let weights_file = open_weights_file(model_path)
            .map_err(|_| GeneratorError::UnableToLoadWeights)?;
        let weights_loader =
            ParameterLoader::new(&weights_file, context.as_ref())
                .map_err(|_| GeneratorError::UnableToLoadWeights)?;
        let weights = weights_loader.tree();

        let backend =
            B::new(context.clone(), &config, &decoder_config, &weights)?;

        let mut generator = Self {
            config,
            tokens: Vec::new(),
            backend,
        };

        //Warmup
        generator.warmup(generator.config.prefill_step_size);
        generator.warmup(generator.config.generate_suffix_length());

        return Ok(generator);
    }
}

impl<B: Backend> Generator<B> {
    pub fn prefill(
        &mut self,
        tokens: Vec<u64>,
        sampling_config: SamplingConfig,
        prefix_offset: usize,
    ) -> PrefillResult {
        assert!(!tokens.is_empty());

        self.tokens.extend(tokens.clone());

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

        let mut last_sampling_output: Option<B::Array> = None;
        let mut run_times: Vec<f64> = Vec::new();

        // Process each prefill step and update the KV cache.
        for step in 0..number_of_prefill_steps {
            let tokens_start_index = step * self.config.prefill_step_size;
            let tokens_end_index =
                tokens_start_index + self.config.prefill_step_size;
            let tokens_for_step =
                &padded_tokens[tokens_start_index..tokens_end_index];
            let positions_for_step =
                &padded_positions[tokens_start_index..tokens_end_index];

            // We need to release the reference to sampling output, because will write into the same buffer.
            let _ = last_sampling_output.take();

            let RunResult {
                sampling_output,
                duration,
            } = self.backend.run(
                tokens_for_step,
                positions_for_step,
                self.config.prefill_step_size,
                Some(sampling_config.clone()),
                false,
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
                    self.backend.accept_tokens(&positions_for_step.as_slice());
                }
            }

            last_sampling_output = sampling_output;
            run_times.push(duration);
        }

        let argmax_tokens = sampling_output_into_vec(&last_sampling_output);

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

        self.backend.accept_tokens(accepted_token_indices.as_slice());

        self.tokens.extend(accepted_tokens.clone());

        PrefillResult {
            tokens: accepted_tokens,
            forwardpass_durations: run_times,
        }
    }

    pub fn generate(
        &mut self,
        sampling_config: SamplingConfig,
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

        let padded_positions: Vec<usize> = speculated_suffix
            .indices
            .iter()
            .map(|idx| idx + start_position)
            .chain(
                std::iter::repeat(INVALID_POSITION).take(unused_tokens_count),
            )
            .collect();

        let RunResult {
            sampling_output,
            duration,
        } = self.backend.run(
            padded_tokens.as_slice(),
            padded_positions.as_slice(),
            1,
            Some(sampling_config),
            false,
        );

        let argmax_tokens = sampling_output_into_vec(&sampling_output);
        drop(sampling_output);

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

        self.backend.accept_tokens(accepted_token_indices.as_slice());

        self.tokens.extend(accepted_tokens.clone());

        GenerateResult {
            tokens: accepted_tokens,
            forwardpass_duration: duration,
        }
    }

    pub fn prefix_length(&self) -> usize {
        self.backend.prefix_length()
    }

    pub fn backend_state(&self) -> B::State {
        self.backend.clone_state()
    }

    pub fn set_backend_state(
        &mut self,
        state: &B::State,
    ) {
        self.backend.restore_state(state);
    }

    pub fn reset_state(&mut self) {
        self.backend.reset_state();
        self.tokens.clear();
    }

    fn warmup(
        &mut self,
        suffix_length: usize,
    ) {
        let _ = self.backend.run(
            vec![0; suffix_length].as_slice(),
            (0..suffix_length).collect::<Vec<usize>>().as_slice(),
            suffix_length,
            None,
            true,
        );
    }
}

fn sampling_output_into_vec<A: Array>(sampling_output: &Option<A>) -> Vec<u64> {
    let buffer = sampling_output
        .as_ref()
        .expect("Sampling output buffer not found - ensure sampling was encoded during forward pass");

    let view = buffer.as_view::<u32>().unwrap();
    let batch_size = buffer.shape()[0];

    let mut result = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        result.push(view[[i]] as u64);
    }

    result
}
