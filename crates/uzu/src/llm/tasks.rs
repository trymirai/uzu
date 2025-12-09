use super::LLMContext;
use crate::backends::metal::forward_pass::{
    EncodableBlock, EncodingParameters, ForwardPassState,
};

pub struct LLMEncodedTask {
    pub key: String,
}

#[derive(Debug, Clone)]
pub struct LLMRunTask {
    pub token_ids: Vec<u64>,
    pub token_positions: Vec<usize>,
    pub token_seeds: Vec<u64>,
    pub expected_number_of_new_tokens: usize,
    pub active_suffix_length: usize,
    pub is_prefilling: bool,
}

impl LLMRunTask {
    pub fn speculate_next_task(&self) -> Self {
        LLMRunTask {
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
        context: &mut LLMContext,
        external_bias_fn: Option<&dyn Fn(usize, usize) -> bool>,
    ) -> ForwardPassState {
        ForwardPassState::new_llm(
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
            external_bias_fn,
        )
    }

    pub fn build_encoded_task(
        &self,
        context: &LLMContext,
        state: &mut ForwardPassState,
        parameters: &EncodingParameters,
        key: String,
    ) -> LLMEncodedTask {
        context.executables.encode(state, &context.command_buffer, parameters);
        LLMEncodedTask {
            key,
        }
    }
}
