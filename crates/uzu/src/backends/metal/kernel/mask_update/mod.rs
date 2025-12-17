use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::backends::metal::{MTLContext, kernel::KernelDataType};

pub struct MaskUpdateKernel {
    pipeline: MTLComputePipelineState,
}

impl MaskUpdateKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, crate::backends::metal::error::MTLError> {
        let function_name = format!(
            "update_attention_mask_{}",
            data_type.function_name_suffix()
        );
        let (pipeline, _) = context
            .compute_pipeline_state_with_reflection(&function_name, None)?;

        Ok(Self {
            pipeline,
        })
    }

    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        mask_buffer: &MTLBuffer,
        unmask_col: i32,
        mask_col: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(mask_buffer), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<i32>() as u64,
            &unmask_col as *const i32 as *const _,
        );
        encoder.set_bytes(
            2,
            std::mem::size_of::<i32>() as u64,
            &mask_col as *const i32 as *const _,
        );
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    }
}
