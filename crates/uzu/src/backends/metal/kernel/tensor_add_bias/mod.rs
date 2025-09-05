use std::mem::size_of;

use metal::{Buffer as MTLBuffer, CommandBuffer as MTLCommandBuffer, ComputePipelineState as MTLComputePipelineState};

use super::{KernelDataType, MTLContext, metal_extensions::ComputeEncoderDispatch};
use crate::backends::metal::error::MTLError;

#[derive(Debug)]
pub struct TensorAddBias {
    pipeline_state: MTLComputePipelineState,
}

impl TensorAddBias {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, MTLError> {
        let function_name = format!("tensorAddBias_{}", data_type.function_name_suffix());
        let (pipeline_state, _reflection) =
            context.compute_pipeline_state_with_reflection(&function_name, None)?;
        Ok(Self { pipeline_state })
    }

    pub fn encode_into_command_buffer(
        &self,
        input: &MTLBuffer,
        bias: &MTLBuffer,
        output: &MTLBuffer,
        num_cols: usize,
        total_len: usize,
        command_buffer: &MTLCommandBuffer,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("Tensor Add Bias");
        encoder.set_compute_pipeline_state(&self.pipeline_state);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(bias), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, size_of::<i32>() as u64, &(num_cols as i32) as *const _ as *const _);
        encoder.set_bytes(4, size_of::<i32>() as u64, &(total_len as i32) as *const _ as *const _);
        encoder.dispatch_1d_exactly(&self.pipeline_state, total_len, None);
        encoder.end_encoding();
    }
}


