use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBuffer as MTLCommandBuffer,
    ComputePipelineState as MTLComputePipelineState,
};

use super::{KernelDataType, MTLContext, metal_extensions::ComputeEncoderDispatch};
use crate::backends::metal::error::MTLError;

#[derive(Debug)]
pub struct TensorAddSwapKernel {
    pipeline_state: MTLComputePipelineState,
}

impl TensorAddSwapKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, MTLError> {
        let function_name =
            format!("tensorAddSwap_{}", data_type.function_name_suffix());

        let (pipeline_state, _argument_names) = context
            .compute_pipeline_state_with_reflection(&function_name, None)?;

        Ok(Self {
            pipeline_state,
        })
    }

    pub fn encode_into_command_buffer(
        &self,
        skip_buffer: &MTLBuffer,
        main_buffer: &MTLBuffer,
        length: usize,
        command_buffer: &MTLCommandBuffer,
    ) {
        let compute_encoder = command_buffer.new_compute_command_encoder();

        compute_encoder.set_label("Tensor Add-Swap");

        compute_encoder.set_buffer(0, Some(skip_buffer), 0);
        compute_encoder.set_buffer(1, Some(main_buffer), 0);

        compute_encoder.set_bytes(
            2,
            size_of::<i32>() as u64,
            &(length as i32) as *const _ as *const std::ffi::c_void,
        );

        compute_encoder.dispatch_1d_exactly(&self.pipeline_state, length, None);

        compute_encoder.end_encoding();
    }
}
