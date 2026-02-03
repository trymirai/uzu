use crate::backends::metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    ProtocolObject,
};

use super::{
    super::{
        MTLContext, MTLError,
        forward_pass::{ArrayId, ForwardPassState},
        kernel::dsl::ActivationMetalKernel,
    },
    EncodableBlock, EncodingParameters,
};
use crate::backends::common::kernel::ActivationKernel;
use crate::{Array, DataType, config::Activation as ActivationConfig};

pub struct Activation {
    kernel: ActivationMetalKernel,
    config: ActivationConfig,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
}

impl Activation {
    pub fn new(
        context: &MTLContext,
        data_type: DataType,
        config: ActivationConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<Self, MTLError> {
        let kernel = ActivationMetalKernel::new(context, data_type.into())?;
        Ok(Self {
            kernel,
            config,
            input_array_id,
            output_array_id,
        })
    }
}

impl EncodableBlock for Activation {
    fn encode(
        &self,
        state: &mut ForwardPassState,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        _parameters: &EncodingParameters,
    ) {
        let encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");

        self.encode_with_shared_encoder(state, &encoder, _parameters);

        encoder.end_encoding();
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
        let arrays = state.arrays(&[self.input_array_id, self.output_array_id]);
        let mut input_array = arrays[0].borrow_mut();
        let mut output_array = arrays[1].borrow_mut();

        let n = input_array.shape().iter().product::<usize>();

        let input_buffer = unsafe { input_array.mtl_buffer() };
        let output_buffer = unsafe { output_array.mtl_buffer() };

        let act_type = match self.config {
            ActivationConfig::SiLU {
                ..
            } => 0,
            ActivationConfig::Gelu => 1,
            ActivationConfig::Identity => {
                panic!("Identity activation is not supported for kernel")
            },
        };

        self.kernel.encode(
            input_buffer,
            output_buffer,
            n as u32,
            act_type,
            encoder,
        );
    }
}
