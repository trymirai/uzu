//! Tensor add-swap encodable.

use metal::CommandBufferRef;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    Array,
    backends::metal::{
        MTLContext, MTLError,
        forward_pass::{ArrayId, ForwardPassState},
        kernel::TensorAddSwapKernel,
    },
};

pub struct TensorAddSwap {
    kernel: TensorAddSwapKernel,
    argument_arrays: Box<[ArrayId]>,
}

impl TensorAddSwap {
    pub fn new(
        context: &MTLContext,
        data_type: crate::backends::metal::KernelDataType,
        argument_arrays: Box<[ArrayId]>,
    ) -> Result<Self, MTLError> {
        let kernel = TensorAddSwapKernel::new(context, data_type)?;
        Ok(Self {
            kernel,
            argument_arrays,
        })
    }
}

impl EncodableBlock for TensorAddSwap {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &CommandBufferRef,
        parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&self.argument_arrays);
        assert_eq!(arrays.len(), 2, "TensorAddSwap expects exactly 2 arrays");

        let length = arrays[0].borrow().num_elements();

        let mut skip_array = arrays[0].borrow_mut();
        let mut main_array = arrays[1].borrow_mut();
        let skip_mtl_buffer = unsafe { skip_array.mtl_buffer() };
        let main_mtl_buffer = unsafe { main_array.mtl_buffer() };

        self.kernel.encode_into_command_buffer(
            &skip_mtl_buffer,
            &main_mtl_buffer,
            length,
            command_buffer,
        );

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }
}
