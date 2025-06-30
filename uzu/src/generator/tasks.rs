use super::context::GeneratorContext;
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
    pub padding_mask: Option<Vec<bool>>,
    pub expected_amount_of_new_tokens: usize,
}

impl GeneratorRunTask {
    pub fn speculate_next_task(&self) -> Self {
        GeneratorRunTask {
            token_ids: self.token_ids.clone(),
            token_positions: self.token_positions.clone(),
            padding_mask: self.padding_mask.clone(),
            expected_amount_of_new_tokens: self.expected_amount_of_new_tokens,
        }
    }

    pub fn encoded_task_key(&self) -> String {
        "".to_string()

        // format!(
        //     "{}-{}"
        //     // use hash map values of window_size -> [suffix length]
        // )
    }

    pub fn create_state(
        &self,
        context: &mut GeneratorContext,
    ) -> ForwardPassState {
        let state = ForwardPassState::new(
            context.mtl_context.clone(),
            &context.model_shape,
            &context.scratch_buffers,
            context.kv_cache.clone(),
            context.shared_buffers.clone(),
            &self.token_ids,
            &self.token_positions,
            self.padding_mask.as_deref(),
            false,
        );

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
}
