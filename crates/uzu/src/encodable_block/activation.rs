//! Activation encodable.

use std::ops::DerefMut;

use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer,
        gpu_types::ActivationType,
        kernel::{ActivationKernel, Kernels},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub struct Activation<B: Backend> {
    kernel: <B::Kernels as Kernels>::ActivationKernel,
    config: ActivationType,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
}

impl<B: Backend> Activation<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        config: ActivationType,
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

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let arrays = state.arrays(&[self.input_array_id, self.output_array_id]);
        let input_array = arrays[0].borrow();
        let output_array = arrays[1].borrow_mut();

        let n = input_array.shape().iter().product::<usize>();
        if self.config == ActivationType::IDENTITY {
            panic!("Identity activation is not supported for kernel");
        }

        let input_buffer = (self.input_array_id != self.output_array_id).then(|| input_array.buffer());
        let input_buffer_borrow = input_buffer.as_ref().map(|b| b.borrow());
        self.kernel.encode(
            input_buffer_borrow.as_deref(),
            output_array.buffer().borrow_mut().deref_mut(),
            n as u32,
            self.config,
            command_buffer,
        );
        Ok(())
    }
}
