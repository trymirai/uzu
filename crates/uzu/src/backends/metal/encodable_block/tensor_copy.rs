//! Tensor copy encodable.

use metal::ComputeCommandEncoderRef;
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    Array,
    backends::metal::{
        MTLContext, MTLError,
        forward_pass::{ArrayId, ForwardPassState},
        kernel::TensorCopyKernel,
    },
};

pub struct TensorCopy {
    kernel: TensorCopyKernel,
    argument_arrays: Box<[ArrayId]>,
}

impl TensorCopy {
    pub fn new(
        context: &MTLContext,
        data_type: crate::backends::metal::KernelDataType,
        argument_arrays: Box<[ArrayId]>,
    ) -> Result<Self, MTLError> {
        let kernel = TensorCopyKernel::new(context, data_type)?;
        Ok(Self {
            kernel,
            argument_arrays,
        })
    }
}

impl EncodableBlock for TensorCopy {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        parameters: &EncodingParameters,
    ) {
        let root = command_buffer.root_command_buffer().to_owned();
        let encoder = root.new_compute_command_encoder();
        self.encode_with_shared_encoder(state, &encoder, parameters);
        encoder.end_encoding();

        if parameters.wait_until_completed {
            command_buffer.commit_and_continue();
            root.wait_until_completed();
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

        self.kernel
            .encode_with_encoder(&source_mtl_buffer, &destination_mtl_buffer, length, encoder);
    }
}
