use metal::ComputeCommandEncoderRef;
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{ArrayId, ForwardPassState, FrozenState};

#[derive(Clone)]
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

    fn supports_shared_encoder(&self) -> bool {
        false
    }

    fn encode_with_shared_encoder(
        &self,
        _state: &mut ForwardPassState,
        _encoder: &ComputeCommandEncoderRef,
        _parameters: &EncodingParameters,
    ) {
        panic!("encode_with_shared_encoder called on unsupported type");
    }

    /// Returns the list of ArrayIds this component needs for encoding.
    /// Used to pre-extract buffers for parallel encoding.
    fn required_buffers(&self) -> Vec<ArrayId> {
        vec![] // Default: no buffers (for backwards compatibility)
    }

    /// Supports parallel encoding with FrozenState
    fn supports_parallel_encode(&self) -> bool {
        false
    }

    /// Encode with a frozen (thread-safe) state snapshot.
    /// Only available if supports_parallel_encode() returns true.
    fn encode_parallel(
        &self,
        _encoder: &ComputeCommandEncoderRef,
        _frozen: &FrozenState,
        _parameters: &EncodingParameters,
    ) {
        panic!("encode_parallel called on unsupported type");
    }
}
