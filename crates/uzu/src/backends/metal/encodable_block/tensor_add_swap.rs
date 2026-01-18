//! Tensor add-swap encodable.

use metal::{CommandBufferRef, ComputeCommandEncoderRef};

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
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_with_shared_encoder(state, &encoder, parameters);
        encoder.end_encoding();

        if parameters.wait_until_completed {
            command_buffer.commit();
            command_buffer.wait_until_completed();
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
        assert_eq!(arrays.len(), 2, "TensorAddSwap expects exactly 2 arrays");

        let length = arrays[0].borrow().num_elements();

        let mut skip_array = arrays[0].borrow_mut();
        let mut main_array = arrays[1].borrow_mut();
        let skip_mtl_buffer = unsafe { skip_array.mtl_buffer() };
        let main_mtl_buffer = unsafe { main_array.mtl_buffer() };

        self.kernel.encode_with_encoder(
            &skip_mtl_buffer,
            &main_mtl_buffer,
            length,
            encoder,
        );
    }
}
