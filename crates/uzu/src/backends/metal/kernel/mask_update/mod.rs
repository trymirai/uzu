use std::ptr::NonNull;

use metal::MTLComputeCommandEncoder;

use crate::backends::metal::{
    ComputeCommandEncoderRef, ComputePipelineState, MTLBuffer, MTLContext,
    MTLSize, ProtocolObject, kernel::KernelDataType,
};

pub struct MaskUpdateKernel {
    pipeline: ComputePipelineState,
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
        encoder: ComputeCommandEncoderRef<'_>,
        mask_buffer: &ProtocolObject<dyn MTLBuffer>,
        unmask_col: i32,
        mask_col: i32,
    ) {
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(Some(mask_buffer), 0, 0);
        unsafe {
            encoder.set_bytes(
                NonNull::new_unchecked(
                    &unmask_col as *const i32 as *mut std::ffi::c_void,
                ),
                std::mem::size_of::<i32>(),
                1,
            );
            encoder.set_bytes(
                NonNull::new_unchecked(
                    &mask_col as *const i32 as *mut std::ffi::c_void,
                ),
                std::mem::size_of::<i32>(),
                2,
            );
        }
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    }
}
