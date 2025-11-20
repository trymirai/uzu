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

    pub fn disable_execution(&self) {
        unsafe {
            let ptr = self.predicate_buffer.contents() as *mut u32;
            *ptr = 1;
            self.predicate_buffer.did_modify_range(metal::NSRange::new(
                0,
                size_of::<u32>() as u64,
            ));
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeneratorRunTask<'a> {
    pub token_ids: &'a [u64],
    pub token_positions: &'a [usize],
    pub token_seeds: &'a [u64],
    pub expected_number_of_new_tokens: usize,
    pub active_suffix_length: usize,
    pub is_prefilling: bool,
}

impl<'a> GeneratorRunTask<'a> {
    pub fn speculate_next_task(&self) -> Self {
        GeneratorRunTask {
            token_ids: self.token_ids,
            token_positions: self.token_positions,
            token_seeds: self.token_seeds,
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
            self.token_ids,
            self.token_positions,
            self.token_seeds,
            self.active_suffix_length,
            self.is_prefilling,
            false,
            external_bias_fn,
            false,
            None,
            None,
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
        context.executables.encode(state, &context.command_buffer, parameters);

        let encoded_task = GeneratorEncodedTask {
            key,
            predicate_buffer: context.mtl_context.device.new_buffer(
                size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            ),
        };

        encoded_task
    }
}
