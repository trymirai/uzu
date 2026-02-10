use crate::{
    backends::common::{Backend, CommandBuffer},
    forward_pass::state::ForwardPassState,
};

mod sampling;

pub use sampling::Sampling;

#[derive(Clone)]
pub struct EncodingParameters<'a, B: Backend> {
    pub warmup: bool,
    pub enable_commit: bool,
    pub wait_until_completed: bool,
    pub projection_step: Option<usize>,
    pub predicate: Option<&'a B::NativeBuffer>,
}

impl<'a, B: Backend> EncodingParameters<'a, B> {
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
        predicate: &'a B::NativeBuffer,
    ) -> Self {
        self.predicate = Some(predicate);
        self
    }

    pub fn predicate_ref(&self) -> Option<&B::NativeBuffer> {
        self.predicate
    }
}

pub trait EncodableBlock<B: Backend> {
    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &B::ComputeEncoder,
        parameters: &EncodingParameters<B>,
    );

    fn supports_shared_encoder(&self) -> bool {
        false
    }

    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        command_buffer: &B::CommandBuffer,
        parameters: &EncodingParameters<B>,
    ) {
        command_buffer.with_compute_encoder(|encoder| self.encode_with_shared_encoder(state, encoder, parameters));
    }
}
