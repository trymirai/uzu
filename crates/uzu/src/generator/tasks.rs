use super::context::GeneratorContext;
use crate::backends::metal::{
    ForwardPassState,
    forward_pass::encodable_with_state::EncodingParameters,
};

pub struct GeneratorEncodedTask {
    pub key: String,
    pub embed_fence: Option<metal::Fence>,
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
        suffix_length: usize,
    ) -> String {
        format!(
            "tokens:{}_suffix:{}",
            tokens_count, suffix_length
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

    /// Encode everything into orchestrator
    pub fn build_encoded_task(
        &self,
        context: &GeneratorContext,
        state: &mut ForwardPassState,
        parameters: &EncodingParameters,
        key: String,
    ) -> GeneratorEncodedTask {
        context.executables.encode_into_orchestrator(
            state,
            &context.orchestrator,
            parameters,
            Some(&context.gpu_sampler),
        );
        GeneratorEncodedTask { key, embed_fence: None }
    }
    
    /// Pre-encode only layers (not embed) - used when embed is MPSGraph-based
    /// Returns the embed_fence that must be signaled when fresh-encoding embed
    pub fn pre_encode_layers_only(
        &self,
        context: &GeneratorContext,
        state: &mut ForwardPassState,
        parameters: &EncodingParameters,
        key: String,
    ) -> GeneratorEncodedTask {
        let embed_fence = context.executables.encode_layers_only(
            state,
            &context.orchestrator,
            parameters,
            Some(&context.gpu_sampler),
        );
        GeneratorEncodedTask { key, embed_fence: Some(embed_fence) }
    }
}
