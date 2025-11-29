use metal::{Buffer, BufferRef, ComputeCommandEncoderRef};
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::ForwardPassState;

#[derive(Clone)]
pub struct EncodingParameters<'a> {
    pub warmup: bool,
    pub enable_commit: bool,
    pub wait_until_completed: bool,
    pub projection_step: Option<usize>,
    pub predicate: Option<&'a Buffer>,
}

impl<'a> EncodingParameters<'a> {
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
            predicate: None,
        }
    }

    pub fn with_projection(
        mut self,
        projection_step: usize,
    ) -> Self {
        self.projection_step = Some(projection_step);
        self
    }

    pub fn with_predicate(
        mut self,
        predicate: &'a Buffer,
    ) -> Self {
        self.predicate = Some(predicate);
        self
    }

    pub fn predicate_ref(&self) -> Option<&BufferRef> {
        self.predicate.map(|buffer| buffer.as_ref())
    }
}

pub trait EncodableWithState {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    );

    fn supports_shared_encoder(&self) -> bool;

    fn encode_with_shared_encoder(
        &self,
        _state: &mut ForwardPassState,
        _encoder: &ComputeCommandEncoderRef,
        _parameters: &EncodingParameters,
    ) {
        panic!("encode_with_shared_encoder called on unsupported type");
    }
}
