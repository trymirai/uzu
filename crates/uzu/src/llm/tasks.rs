use std::mem::size_of;

use metal::{Buffer, MTLResourceOptions};

use super::LanguageModelGeneratorContext;
use crate::backends::metal::forward_pass::{
    EncodableBlock, EncodingParameters, ForwardPassState,
};

pub struct LanguageModelGeneratorEncodedTask {
    pub key: String,
    predicate_buffer: Buffer,
}

impl LanguageModelGeneratorEncodedTask {
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
pub struct LanguageModelGeneratorRunTask<'a> {
    pub token_ids: &'a [u64],
    pub token_positions: &'a [usize],
    pub token_bitmask: Option<&'a [u32]>,
    pub token_seeds: &'a [u64],
    pub expected_number_of_new_tokens: usize,
    pub active_suffix_length: usize,
    pub is_prefilling: bool,
}

impl<'a> LanguageModelGeneratorRunTask<'a> {
    pub fn speculate_next_task(&self) -> Self {
        LanguageModelGeneratorRunTask {
            token_ids: self.token_ids,
            token_positions: self.token_positions,
            token_bitmask: self.token_bitmask,
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
        context: &mut LanguageModelGeneratorContext,
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
    ) -> ForwardPassState {
        ForwardPassState::new_llm(
            context.mtl_context.clone(),
            &context.model_config.decoder_config,
            &context.model_shape,
            &context.scratch_buffers,
            context.cache_layers.clone(),
            context.shared_buffers.clone(),
            self.token_ids,
            self.token_positions,
            self.token_bitmask,
            self.token_seeds,
            self.active_suffix_length,
            self.is_prefilling,
            external_bias_fn,
            false,
            None,
            None,
        )
    }

    pub fn build_encoded_task(
        &self,
        context: &LanguageModelGeneratorContext,
        state: &mut ForwardPassState,
        parameters: &EncodingParameters,
        key: String,
    ) -> LanguageModelGeneratorEncodedTask {
        context.executables.encode(state, &context.command_buffer, parameters);

        LanguageModelGeneratorEncodedTask {
            key,
            predicate_buffer: context.mtl_context.device.new_buffer(
                size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            ),
        }
    }
}
