//! Tensor add-swap encodable.

use crate::backends::{
    common::kernel::TensorAddSwapKernel,
    metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
        ProtocolObject, Retained,
    },
};

use super::{EncodableBlock, EncodingParameters, Metal};
use crate::backends::metal::{
    MTLContext, MTLError,
    forward_pass::{ArrayId, ForwardPassState},
    kernel::dsl::TensorAddSwapMetalKernel,
};

pub struct TensorAddSwap {
    kernel: TensorAddSwapMetalKernel,
    argument_arrays: Box<[ArrayId]>,
}

impl TensorAddSwap {
    pub fn new(
        context: &MTLContext,
        data_type: crate::backends::metal::KernelDataType,
        argument_arrays: Box<[ArrayId]>,
    ) -> Result<Self, MTLError> {
        let kernel = TensorAddSwapMetalKernel::new(context, data_type.into())?;
        Ok(Self {
            kernel,
            argument_arrays,
        })
    }
}

impl EncodableBlock<Metal> for TensorAddSwap {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        parameters: &EncodingParameters,
    ) {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
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
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        _parameters: &EncodingParameters,
    ) {
        let arrays = state.arrays(&self.argument_arrays);
        assert_eq!(arrays.len(), 2, "TensorAddSwap expects exactly 2 arrays");

        let length = arrays[0].borrow().num_elements();

        let skip_array = arrays[0].borrow_mut();
        let main_array = arrays[1].borrow_mut();
        let skip_mtl_buffer = skip_array.buffer();
        let main_mtl_buffer = main_array.buffer();

        self.kernel.encode(
            &skip_mtl_buffer,
            &main_mtl_buffer,
            length as u32,
            encoder,
        );
    }
}
