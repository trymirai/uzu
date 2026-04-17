//! Tensor copy encodable.

use std::ops::{Deref, DerefMut};

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder,
        kernel::{Kernels, TensorCopyKernel},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub struct TensorCopy<B: Backend> {
    kernel: <B::Kernels as Kernels>::TensorCopyKernel,
    source_array: ArrayId,
    destination_array: ArrayId,
}

impl<B: Backend> TensorCopy<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        source_array: ArrayId,
        destination_array: ArrayId,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::TensorCopyKernel::new(context, data_type)?;

        Ok(Self {
            kernel,
            source_array,
            destination_array,
        })
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let source_array = state.array(self.source_array);
        let destination_array = state.array(self.destination_array);

        let length = source_array.num_elements();

        self.kernel.encode(
            source_array.buffer().borrow().deref(),
            destination_array.buffer().borrow_mut().deref_mut(),
            length as u32,
            encoder,
        );
        Ok(())
    }
}
