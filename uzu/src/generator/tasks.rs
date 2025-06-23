use super::{
    context::GeneratorContext,
    mask_descriptor::{GeneratorMaskDescriptor, GeneratorPrefixLength},
};
use crate::backends::metal::{
    ForwardPassState,
    forward_pass::encodable_with_state::{
        EncodableWithState, EncodingParameters,
    },
};

pub struct GeneratorEncodedTask {
    pub key: String,
}

#[derive(Debug, Clone)]
pub struct GeneratorRunTask {
    pub token_ids: Vec<u64>,
    pub token_positions: Vec<usize>,
    pub mask_descriptor: GeneratorMaskDescriptor,
    pub expected_amount_of_new_tokens: usize,
}

impl GeneratorRunTask {
    pub fn speculate_next_task(&self) -> Self {
        GeneratorRunTask {
            token_ids: self.token_ids.clone(),
            token_positions: self.token_positions.clone(),
            mask_descriptor: GeneratorMaskDescriptor {
                suffix_length: self.mask_descriptor.suffix_length,
                prefix_length: GeneratorPrefixLength {
                    real: self.mask_descriptor.prefix_length.real
                        + self.expected_amount_of_new_tokens,
                    step: self.mask_descriptor.prefix_length.step,
                },
                casual_mask: self.mask_descriptor.casual_mask.clone(),
            },
            expected_amount_of_new_tokens: self.expected_amount_of_new_tokens,
        }
    }

    pub fn encoded_task_key(&self) -> String {
        format!(
            "{}-{}",
            self.mask_descriptor.suffix_length,
            self.mask_descriptor.prefix_length.padded()
        )
    }

    pub fn create_state(
        &self,
        context: &mut GeneratorContext,
    ) -> ForwardPassState {
        self.apply_to_context(context);
        let mut state = ForwardPassState::new(
            context.mtl_context.clone(),
            &context.model_shape,
            &context.scratch_buffers,
            context.kv_cache.clone(),
            context.shared_buffers.clone(),
            &self.token_ids,
            &self.token_positions,
            {
                let effective_prefix_length =
                    context.kv_cache.borrow().max_effective_prefix_length();
                self.mask_descriptor.create_attention_bias_closure_with_prefix(
                    effective_prefix_length,
                )
            },
            false,
        );
        self.apply_to_state(&mut state);
        return state;
    }

    pub fn build_encoded_task(
        &self,
        context: &GeneratorContext,
        state: &mut ForwardPassState,
        parameters: &EncodingParameters,
    ) -> GeneratorEncodedTask {
        context.executables.encode(state, &context.command_buffer, parameters);
        GeneratorEncodedTask {
            key: self.encoded_task_key(),
        }
    }

    pub fn apply_to_context(
        &self,
        context: &mut GeneratorContext,
    ) {
        let total_context_length =
            context.kv_cache.borrow().max_prefix_length()
                + context.kv_cache.borrow().max_suffix_length();

        let window = context
            .model_shape
            .sliding_window_length_per_layer
            .iter()
            .filter_map(|o| *o)
            .filter(|&window_size| window_size < total_context_length)
            .next()
            .unwrap_or(usize::MAX);

        let effective_prefix_real =
            self.mask_descriptor.prefix_length.real.min(window);

        let effective_prefix_padded =
            if let Some(step) = self.mask_descriptor.prefix_length.step {
                let remainder = effective_prefix_real % step;
                effective_prefix_real
                    + if remainder == 0 {
                        0
                    } else {
                        step - remainder
                    }
            } else {
                effective_prefix_real
            };

        context
            .kv_cache
            .borrow_mut()
            .set_prefix_length(effective_prefix_padded);
    }

    fn apply_to_state(
        &self,
        state: &mut ForwardPassState,
    ) {
        if let Some(kv_cache_update_task) =
            self.mask_descriptor.kv_cache_update_task()
        {
            state.kv_cache_update_source_indices =
                kv_cache_update_task.sources_indices;
            state.kv_cache_update_destination_indices =
                kv_cache_update_task.destination_indices;
        }
    }
}
