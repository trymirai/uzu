use std::mem::size_of;

use metal::{MTLCommandBuffer, MTLCommandEncoder};
use objc2::msg_send;
use objc2_foundation::NSString;

use crate::backends::metal::{
    BufferRef, CommandBufferRef, ComputeCommandEncoderRef,
    ComputeEncoderLegacy, ComputePipelineState, KernelDataType, MTLContext,
    MTLError,
    BufferLabelExt,
    metal_extensions::{
        ComputeEncoderConditional, ComputeEncoderDispatch,
    },
};

#[derive(Debug)]
pub struct TensorAddBias {
    pipeline_state: ComputePipelineState,
}

impl TensorAddBias {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, MTLError> {
        let function_name =
            format!("tensorAddBias_{}", data_type.function_name_suffix());
        let (pipeline_state, _reflection) = context
            .compute_pipeline_state_with_reflection(&function_name, None)?;
        Ok(Self {
            pipeline_state,
        })
    }

    pub fn encode_into_command_buffer(
        &self,
        input: BufferRef<'_>,
        bias: BufferRef<'_>,
        output: BufferRef<'_>,
        num_cols: usize,
        total_len: usize,
        command_buffer: CommandBufferRef<'_>,
        predicate: Option<BufferRef<'_>>,
    ) {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        self.encode_with_encoder(
            input, bias, output, num_cols, total_len, &encoder, predicate,
        );
        encoder.end_encoding();
    }

    pub fn encode_with_encoder(
        &self,
        input: BufferRef<'_>,
        bias: BufferRef<'_>,
        output: BufferRef<'_>,
        num_cols: usize,
        total_len: usize,
        encoder: ComputeCommandEncoderRef<'_>,
        predicate: Option<BufferRef<'_>>,
    ) {
        encoder.condition(
            predicate,
            || {
                unsafe {
                    let label = NSString::from_str("Tensor Add Bias");
                    let _: () = msg_send![encoder, setLabel: &*label];
                }
                encoder.set_compute_pipeline_state(&self.pipeline_state);
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(bias), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(
                    3,
                    size_of::<i32>() as u64,
                    &(num_cols as i32) as *const _ as *const _,
                );
                encoder.set_bytes(
                    4,
                    size_of::<i32>() as u64,
                    &(total_len as i32) as *const _ as *const _,
                );
                encoder.dispatch_1d_exactly(
                    &self.pipeline_state,
                    total_len,
                    None,
                );
            },
            None::<fn()>,
        );
    }
}
