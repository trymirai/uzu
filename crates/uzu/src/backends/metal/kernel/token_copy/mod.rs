use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::backends::metal::MTLContext;

pub struct TokenCopyKernel {
    pipeline: MTLComputePipelineState,
}

impl TokenCopyKernel {
    pub fn new(
        context: &MTLContext,
    ) -> Result<Self, crate::backends::metal::error::MTLError> {
        let (pipeline, _) = context
            .compute_pipeline_state_with_reflection("copy_sampled_token", None)?;

        Ok(Self { pipeline })
    }

    pub fn encode(
        &self,
        src: &MTLBuffer,
        dst: &MTLBuffer,
        encoder: &ComputeCommandEncoderRef,
    ) {
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(src), 0);
        encoder.set_buffer(1, Some(dst), 0);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    }

    pub fn encode_to_offset(
        &self,
        src: &MTLBuffer,
        dst: &MTLBuffer,
        dst_offset: usize,
        encoder: &ComputeCommandEncoderRef,
    ) {
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(src), 0);
        encoder.set_buffer(1, Some(dst), (dst_offset * std::mem::size_of::<u32>()) as u64);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    }
}

