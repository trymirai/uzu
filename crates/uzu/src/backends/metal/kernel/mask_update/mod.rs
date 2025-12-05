use metal::{
    Buffer as MTLBuffer, ComputeCommandEncoderRef,
    ComputePipelineState as MTLComputePipelineState, MTLSize,
};

use crate::backends::metal::{MTLContext, kernel::KernelDataType};

/// Kernel for updating attention mask between async passes.
pub struct MaskUpdateKernel {
    pipeline: MTLComputePipelineState,
    data_type: KernelDataType,
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

        Ok(Self { pipeline, data_type })
    }

    /// Encodes mask update operation.
    /// - unmask_col: column to set to 0 (new KV position)
    /// - mask_col: column to set to -inf (evicted position), -1 if none
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

        // Set unmask_value = 0.0 and mask_value = -inf based on data type
        match self.data_type {
            KernelDataType::Float32 => {
                let unmask: f32 = 0.0;
                let mask: f32 = f32::NEG_INFINITY;
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<f32>() as u64,
                    &unmask as *const f32 as *const _,
                );
                encoder.set_bytes(
                    4,
                    std::mem::size_of::<f32>() as u64,
                    &mask as *const f32 as *const _,
                );
            }
            KernelDataType::Float16 | KernelDataType::BFloat16 => {
                // For f16/bf16, we pass as u16 bit pattern
                let unmask: u16 = 0; // 0.0 in f16/bf16
                let mask: u16 = if matches!(self.data_type, KernelDataType::Float16) {
                    0xFC00 // -inf in f16
                } else {
                    0xFF80 // -inf in bf16
                };
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<u16>() as u64,
                    &unmask as *const u16 as *const _,
                );
                encoder.set_bytes(
                    4,
                    std::mem::size_of::<u16>() as u64,
                    &mask as *const u16 as *const _,
                );
            }
        }

        // Single thread is enough for this simple operation
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    }
}



