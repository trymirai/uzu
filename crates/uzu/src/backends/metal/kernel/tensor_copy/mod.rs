use std::mem::size_of;

use metal::{
    Buffer as MTLBuffer, CommandBuffer as MTLCommandBuffer,
    ComputeCommandEncoderRef, ComputePipelineState as MTLComputePipelineState,
};
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{
    KernelDataType, MTLContext, metal_extensions::ComputeEncoderDispatch,
};
use crate::{
    Array,
    backends::metal::forward_pass::{
        ArrayId, ForwardPassState,
        encodable_with_state::{EncodableWithState, EncodingParameters},
    },
};

pub struct TensorCopy {
    pipeline_state: MTLComputePipelineState,
    argument_arrays: Box<[ArrayId]>,
}

impl TensorCopy {
    pub fn new(
        context: &MTLContext,
        data_type: KernelDataType,
        argument_arrays: Box<[ArrayId]>,
    ) -> Result<Self, crate::backends::metal::error::MTLError> {
        let function_name =
            format!("tensorCopy_{}", data_type.function_name_suffix());

        let (pipeline_state, _argument_names) = context
            .compute_pipeline_state_with_reflection(&function_name, None)?;

        Ok(Self {
            pipeline_state,
            argument_arrays,
        })
    }

    fn encode_into_command_buffer(
        &self,
        source_buffer: &MTLBuffer,
        destination_buffer: &MTLBuffer,
        length: usize,
        command_buffer: &MTLCommandBuffer,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_with_encoder_raw(
            source_buffer,
            destination_buffer,
            length,
            encoder,
        );
        encoder.end_encoding();
    }

    fn encode_with_encoder_raw(
        &self,
        source_buffer: &MTLBuffer,
        destination_buffer: &MTLBuffer,
        length: usize,
        encoder: &ComputeCommandEncoderRef,
    ) {
        encoder.set_label("Tensor Copy");
        encoder.set_buffer(0, Some(source_buffer), 0);
        encoder.set_buffer(1, Some(destination_buffer), 0);
        encoder.set_bytes(
            2,
            size_of::<i32>() as u64,
            &(length as i32) as *const _ as *const std::ffi::c_void,
        );
        encoder.dispatch_1d_exactly(&self.pipeline_state, length, None);
    }
}

impl EncodableWithState for TensorCopy {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&self.argument_arrays);
        assert_eq!(arrays.len(), 2, "TensorCopy expects exactly 2 arrays");

        let length = arrays[0].borrow().num_elements();

        let mut source_array = arrays[0].borrow_mut();
        let mut destination_array = arrays[1].borrow_mut();
        let source_mtl_buffer = unsafe { source_array.mtl_buffer() };
        let destination_mtl_buffer = unsafe { destination_array.mtl_buffer() };

        let retained_mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();

        self.encode_into_command_buffer(
            &source_mtl_buffer,
            &destination_mtl_buffer,
            length,
            &retained_mtl_command_buffer,
        );

        if parameters.wait_until_completed {
            command_buffer.commit_and_continue();
            retained_mtl_command_buffer.wait_until_completed();
        }
    }

    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState,
        encoder: &ComputeCommandEncoderRef,
        _parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&self.argument_arrays);
        assert_eq!(arrays.len(), 2, "TensorCopy expects exactly 2 arrays");

        let length = arrays[0].borrow().num_elements();

        let mut source_array = arrays[0].borrow_mut();
        let mut destination_array = arrays[1].borrow_mut();
        let source_mtl_buffer = unsafe { source_array.mtl_buffer() };
        let destination_mtl_buffer = unsafe { destination_array.mtl_buffer() };

        self.encode_with_encoder_raw(
            &source_mtl_buffer,
            &destination_mtl_buffer,
            length,
            encoder,
        );
    }
}
