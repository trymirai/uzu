use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::ForwardPassState;

pub struct EncodingParameters {
    pub warmup: bool,
    pub enable_commit: bool,
    pub wait_until_completed: bool,
    pub projection_step: Option<usize>,
}

impl EncodingParameters {
    pub fn new(
        warmup: bool,
        enable_commit: bool,
        wait_until_completed: bool,
    ) -> Self {
        Self {
            warmup,
            enable_commit,
            wait_until_completed,
            projection_step: None,
        }
    }

    pub fn with_projection(
        mut self,
        projection_step: usize,
    ) -> Self {
        self.projection_step = Some(projection_step);
        self
    }
}

pub trait EncodableWithState {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    );
}
