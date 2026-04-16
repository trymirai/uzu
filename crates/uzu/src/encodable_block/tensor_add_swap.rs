//! Tensor add-swap encodable.

use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, TensorAddSwapKernel},
    },
};

pub struct TensorAddSwap<B: Backend> {
    kernel: <B::Kernels as Kernels>::TensorAddSwapKernel,
}

impl<B: Backend> TensorAddSwap<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::TensorAddSwapKernel::new(context, data_type)?;

        Ok(Self {
            kernel,
        })
    }

    pub fn encode(
        &self,
        skip: &mut Allocation<B>,
        main: &mut Allocation<B>,
        length: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.kernel.encode(skip, main, length as u32, encoder);
        Ok(())
    }
}
