//! Tensor copy encodable.

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
        let arrays = state.arrays(&self.argument_arrays);
        assert_eq!(arrays.len(), 2, "TensorCopy expects exactly 2 arrays");

        let length = arrays[0].borrow().num_elements();

        let mut source_array = arrays[0].borrow_mut();
        let mut destination_array = arrays[1].borrow_mut();
        let source_mtl_buffer = unsafe { source_array.mtl_buffer() };
        let destination_mtl_buffer = unsafe { destination_array.mtl_buffer() };

        let retained_mtl_command_buffer =
            command_buffer.root_command_buffer().to_owned();

        self.kernel.encode_into_command_buffer(
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
}
