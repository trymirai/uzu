use std::{mem::size_of, ptr::NonNull};

use metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};
use objc2::msg_send;
use objc2_foundation::NSString;

use crate::backends::metal::{
    ComputeCommandEncoderRef, ComputePipelineState,
    KernelDataType, MTLBuffer, MTLContext, MTLError, ProtocolObject,
    metal_extensions::{ComputeEncoderConditional, ComputeEncoderDispatch},
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
        input: &ProtocolObject<dyn MTLBuffer>,
        bias: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        num_cols: usize,
        total_len: usize,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        predicate: Option<&ProtocolObject<dyn MTLBuffer>>,
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
        input: &ProtocolObject<dyn MTLBuffer>,
        bias: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        num_cols: usize,
        total_len: usize,
        encoder: ComputeCommandEncoderRef<'_>,
        predicate: Option<&ProtocolObject<dyn MTLBuffer>>,
    ) {
        encoder.condition(
            predicate,
            || {
                unsafe {
                    let label = NSString::from_str("Tensor Add Bias");
                    let _: () = msg_send![encoder, setLabel: &*label];
                }
                encoder.set_compute_pipeline_state(&self.pipeline_state);
                encoder.set_buffer(Some(input), 0, 0);
                encoder.set_buffer(Some(bias), 0, 1);
                encoder.set_buffer(Some(output), 0, 2);
                unsafe {
                    encoder.set_bytes(
                        NonNull::new(
                            &(num_cols as i32) as *const i32
                                as *mut std::ffi::c_void,
                        )
                        .unwrap(),
                        size_of::<i32>(),
                        3,
                    );
                    encoder.set_bytes(
                        NonNull::new(
                            &(total_len as i32) as *const i32
                                as *mut std::ffi::c_void,
                        )
                        .unwrap(),
                        size_of::<i32>(),
                        4,
                    );
                }
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
