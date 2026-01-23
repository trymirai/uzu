use std::{mem::size_of, ptr::NonNull};

use metal::MTLComputeCommandEncoder;
use objc2::msg_send;
use objc2_foundation::NSString;

use crate::backends::metal::{
    ComputePipelineState,
    KernelDataType, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLContext,
    MTLError, ProtocolObject, metal_extensions::ComputeEncoderDispatch,
};

pub struct TensorCopyKernel {
    pipeline_state: ComputePipelineState,
}

impl TensorCopyKernel {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
    ) -> Result<Self, MTLError> {
        let function_name =
            format!("tensorCopy_{}", data_type.function_name_suffix());

        let (pipeline_state, _argument_names) = context
            .compute_pipeline_state_with_reflection(&function_name, None)?;

        Ok(Self {
            pipeline_state,
        })
    }

    pub fn encode_into_command_buffer(
        &self,
        source_buffer: &ProtocolObject<dyn MTLBuffer>,
        destination_buffer: &ProtocolObject<dyn MTLBuffer>,
        length: usize,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) {
        let compute_encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        self.encode_with_encoder(
            source_buffer,
            destination_buffer,
            length,
            &compute_encoder,
        );
        compute_encoder.end_encoding();
    }

    pub fn encode_with_encoder(
        &self,
        source_buffer: &ProtocolObject<dyn MTLBuffer>,
        destination_buffer: &ProtocolObject<dyn MTLBuffer>,
        length: usize,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) {
        unsafe {
            let label = NSString::from_str("Tensor Copy");
            let _: () = msg_send![compute_encoder, setLabel: &*label];
        }

        compute_encoder.set_buffer(Some(source_buffer), 0, 0);
        compute_encoder.set_buffer(Some(destination_buffer), 0, 1);
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new_unchecked(
                    &(length as i32) as *const i32 as *mut std::ffi::c_void,
                ),
                size_of::<i32>(),
                2,
            );
        }

        compute_encoder.dispatch_1d_exactly(&self.pipeline_state, length, None);
    }
}
