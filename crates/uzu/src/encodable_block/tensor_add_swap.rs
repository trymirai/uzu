//! Tensor add-swap encodable.

use std::ops::DerefMut;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer,
        kernel::{Kernels, TensorAddSwapKernel},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub struct TensorAddSwap<B: Backend> {
    kernel: <B::Kernels as Kernels>::TensorAddSwapKernel,
    argument_arrays: Box<[ArrayId]>,
}

impl<B: Backend> TensorAddSwap<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        argument_arrays: Box<[ArrayId]>,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::TensorAddSwapKernel::new(context, data_type)?;
        Ok(Self {
            kernel,
            argument_arrays,
        })
    }
}

impl<B: Backend> EncodableBlock<B> for TensorAddSwap<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let arrays = state.arrays(&self.argument_arrays);
        assert_eq!(arrays.len(), 2, "TensorAddSwap expects exactly 2 arrays");

        let length = arrays[0].borrow().num_elements();

        let skip_array = arrays[0].borrow_mut();
        let main_array = arrays[1].borrow_mut();

        self.kernel.encode(
            skip_array.buffer().borrow_mut().deref_mut(),
            main_array.buffer().borrow_mut().deref_mut(),
            length as u32,
            command_buffer,
        );
        Ok(())
    }
}
