use crate::backends::metal::{
    BufferRef, ComputeCommandEncoderRef, ComputeEncoderLegacy,
    ComputePipelineState, MTLContext, MTLError, MTLSize, mtl_size,
};

/// Kernel for copying sampled tokens in async pipeline.
/// Two operations:
/// 1. copy_to_token_ids: u32 → u64 for next forward pass embedding
/// 2. copy_to_results: u32 → u32[offset] for callback to read
pub struct TokenCopyKernel {
    copy_to_token_ids: ComputePipelineState,
    copy_to_results: ComputePipelineState,
}

impl TokenCopyKernel {
    pub fn new(context: &MTLContext) -> Result<Self, MTLError> {
        let (copy_to_token_ids, _) = context
            .compute_pipeline_state_with_reflection("copy_sampled_token", None)?;
        let (copy_to_results, _) = context
            .compute_pipeline_state_with_reflection("copy_token_to_results", None)?;

        Ok(Self {
            copy_to_token_ids,
            copy_to_results,
        })
    }

    /// Copies sampling_output[0] (u32) → token_ids[0] (u64)
    /// For next forward pass to read the token.
    pub fn encode_to_token_ids(
        &self,
        sampling_output: BufferRef<'_>,
        token_ids: BufferRef<'_>,
        encoder: ComputeCommandEncoderRef<'_>,
    ) {
        encoder.set_compute_pipeline_state(&self.copy_to_token_ids);
        encoder.set_buffer(0, Some(sampling_output), 0);
        encoder.set_buffer(1, Some(token_ids), 0);
        encoder.dispatch_threads(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
    }

    /// Copies sampling_output[0] (u32) → results[offset] (u32)
    /// For callback to read without race condition.
    pub fn encode_to_results(
        &self,
        sampling_output: BufferRef<'_>,
        results: BufferRef<'_>,
        pass_idx: usize,
        encoder: ComputeCommandEncoderRef<'_>,
    ) {
        let offset = (pass_idx * std::mem::size_of::<u32>()) as u64;
        encoder.set_compute_pipeline_state(&self.copy_to_results);
        encoder.set_buffer(0, Some(sampling_output), 0);
        encoder.set_buffer(1, Some(results), offset);
        encoder.dispatch_threads(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
    }
}
