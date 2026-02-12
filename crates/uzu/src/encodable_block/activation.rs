//! Activation encodable.

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::{ActivationKernel, Kernels},
    },
    config::Activation as ActivationConfig,
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub struct Activation<B: Backend> {
    kernel: <B::Kernels as Kernels>::ActivationKernel,
    config: ActivationConfig,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
}

impl<B: Backend> Activation<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        config: ActivationConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::ActivationKernel::new(context, data_type)?;
        Ok(Self {
            kernel,
            config,
            input_array_id,
            output_array_id,
        })
    }
}

impl<B: Backend> EncodableBlock<B> for Activation<B> {
    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
    ) {
        let arrays = state.arrays(&[self.input_array_id, self.output_array_id]);
        let input_array = arrays[0].borrow_mut();
        let output_array = arrays[1].borrow_mut();

        let n = input_array.shape().iter().product::<usize>();
        let act_type = match self.config {
            ActivationConfig::SiLU {
                ..
            } => 0,
            ActivationConfig::Gelu => 1,
            ActivationConfig::Identity => panic!("Identity activation is not supported for kernel"),
        };

        self.kernel.encode(input_array.buffer(), output_array.buffer(), n as u32, act_type, encoder);
    }
}
