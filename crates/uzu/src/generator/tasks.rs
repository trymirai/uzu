use std::mem::size_of;

use metal::{Buffer, MTLResourceOptions};

use super::context::GeneratorContext;
use crate::backends::metal::{
    ForwardPassState,
    forward_pass::encodable_with_state::{
        EncodableWithState, EncodingParameters,
    },
};

pub struct GeneratorEncodedTask {
    pub key: String,
    predicate_buffer: Buffer,
}

impl GeneratorEncodedTask {
    pub fn predicate_buffer(&self) -> &Buffer {
        &self.predicate_buffer
    }

    pub fn set_predicate_enabled(
        &self,
        enabled: bool,
    ) {
        unsafe {
            let ptr = self.predicate_buffer.contents() as *mut u32;
            *ptr = u32::from(enabled);
            self.predicate_buffer.did_modify_range(metal::NSRange::new(
                0,
                size_of::<u32>() as u64,
            ));
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeneratorRunTask {
    pub token_ids: Vec<u64>,
    pub token_positions: Vec<usize>,
    pub token_seeds: Vec<u64>,
    pub expected_number_of_new_tokens: usize,
    pub active_suffix_length: usize,
    pub is_prefilling: bool,
}

impl GeneratorRunTask {
    pub fn speculate_next_task(&self) -> Self {
        GeneratorRunTask {
            token_ids: self.token_ids.clone(),
            token_positions: self.token_positions.clone(),
            token_seeds: self.token_seeds.clone(),
            expected_number_of_new_tokens: self.expected_number_of_new_tokens,
            active_suffix_length: self.active_suffix_length,
            is_prefilling: self.is_prefilling,
        }
    }

    pub fn encoded_task_key(
        &self,
        tokens_count: usize,
    ) -> String {
        format!(
            "tokens:{}_suffix:{}",
            tokens_count, self.expected_number_of_new_tokens
        )
    }

    pub fn create_state(
        &self,
        context: &mut GeneratorContext,
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
    ) -> ForwardPassState {
        let state = ForwardPassState::new(
            context.mtl_context.clone(),
            &context.model_config.decoder_config,
            &context.model_shape,
            &context.scratch_buffers,
            context.cache_layers.clone(),
            context.shared_buffers.clone(),
            &self.token_ids,
            &self.token_positions,
            self.active_suffix_length,
            self.is_prefilling,
            &self.token_seeds,
            false,
            external_bias_fn,
        );

        return state;
    }

    pub fn build_encoded_task(
        &self,
        context: &GeneratorContext,
        state: &mut ForwardPassState,
        parameters: &EncodingParameters,
        key: String,
    ) -> GeneratorEncodedTask {
        // TODO: Re-enable predicate buffer when we can ensure consistent visibility
        // and MPSGraph support. For now, we rely on reset_command_buffer.
        // let enabled_value: u32 = 1;
        // let predicate_buffer = context.mtl_context.device.new_buffer_with_data(
        //     &enabled_value as *const u32 as *const _,
        //     size_of::<u32>() as u64,
        //     MTLResourceOptions::StorageModeShared,
        // );

        // let parameters_with_predicate =
        //     parameters.clone().with_predicate(&predicate_buffer);
        context.executables.encode(
            state,
            &context.command_buffer,
            parameters, // Pass original parameters (predicate=None)
        );

        let encoded_task = GeneratorEncodedTask {
            key,
            // Create a dummy buffer or handle Option in GeneratorEncodedTask?
            // GeneratorEncodedTask expects a buffer. I'll create a dummy one but not use it for encoding.
            predicate_buffer: context.mtl_context.device.new_buffer(
                size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            ),
        };
        // encoded_task.set_predicate_enabled(true);
        encoded_task
    }
}
