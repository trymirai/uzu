//! Activation encodable.

use std::ops::DerefMut;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer,
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
        let kernel =
            <B::Kernels as Kernels>::ActivationKernel::new(context, data_type, input_array_id == output_array_id)?;
        Ok(Self {
            kernel,
            config,
            input_array_id,
            output_array_id,
        })
    }
}

impl<B: Backend> EncodableBlock<B> for Activation<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let arrays = state.arrays(&[self.input_array_id, self.output_array_id]);
        let input_array = arrays[0].borrow();
        let output_array = arrays[1].borrow_mut();

        let n = input_array.shape().iter().product::<usize>();
        let act_type = match self.config {
            ActivationConfig::SiLU {
                ..
            } => 0,
            ActivationConfig::Gelu => 1,
            ActivationConfig::Identity => panic!("Identity activation is not supported for kernel"),
        };

        let input_buffer = (self.input_array_id != self.output_array_id).then(|| input_array.buffer());
        let input_buffer_borrow = input_buffer.as_ref().map(|b| b.borrow());
        self.kernel.encode(
            input_buffer_borrow.as_deref(),
            output_array.buffer().borrow_mut().deref_mut(),
            n as u32,
            act_type,
            command_buffer,
        );
        Ok(())
    }
}
