//! Tensor copy encodable.

use crate::backends::{
    common::kernel::TensorCopyKernel,
    metal::{
        MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
        ProtocolObject, Retained,
    },
};

use super::{EncodableBlock, Metal};
use crate::backends::metal::{
    MTLContext, MTLError, kernel::dsl::TensorCopyMetalKernel,
};
use crate::encodable_block::EncodingParameters;
use crate::forward_pass::state::{ArrayId, ForwardPassState};

pub struct TensorCopy {
    kernel: TensorCopyMetalKernel,
    argument_arrays: Box<[ArrayId]>,
}

impl TensorCopy {
    pub fn new(
        context: &MTLContext,
        data_type: crate::backends::metal::KernelDataType,
        argument_arrays: Box<[ArrayId]>,
    ) -> Result<Self, MTLError> {
        let kernel = TensorCopyMetalKernel::new(context, data_type.into())?;
        Ok(Self {
            kernel,
            argument_arrays,
        })
    }
}

impl EncodableBlock<Metal> for TensorCopy {
    fn encode(
        &self,
        state: &mut ForwardPassState<Metal>,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        parameters: &EncodingParameters<Metal>,
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
        state: &mut ForwardPassState<Metal>,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        _parameters: &EncodingParameters<Metal>,
    ) {
        let arrays = state.arrays(&self.argument_arrays);
        assert_eq!(arrays.len(), 2, "TensorCopy expects exactly 2 arrays");

        let length = arrays[0].borrow().num_elements();

        let source_array = arrays[0].borrow_mut();
        let destination_array = arrays[1].borrow_mut();
        let source_mtl_buffer = source_array.buffer();
        let destination_mtl_buffer = destination_array.buffer();

        self.kernel.encode(
            &source_mtl_buffer,
            &destination_mtl_buffer,
            length as u32,
            encoder,
        );
    }
}
