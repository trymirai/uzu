//! Tensor copy encodable.

use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, TensorCopyKernel},
    },
};

pub struct TensorCopy<B: Backend> {
    kernel: <B::Kernels as Kernels>::TensorCopyKernel,
}

impl<B: Backend> TensorCopy<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::TensorCopyKernel::new(context, data_type)?;

        Ok(Self {
            kernel,
        })
    }

    pub fn encode(
        &self,
        source: &Allocation<B>,
        destination: &mut Allocation<B>,
        length: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.kernel.encode(source, destination, length as u32, encoder);
        Ok(())
    }
}
