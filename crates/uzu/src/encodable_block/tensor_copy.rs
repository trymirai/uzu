//! Tensor copy encodable.

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::{Kernels, TensorCopyKernel},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub struct TensorCopy<B: Backend> {
    kernel: <B::Kernels as Kernels>::TensorCopyKernel,
    argument_arrays: Box<[ArrayId]>,
}

impl<B: Backend> TensorCopy<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        argument_arrays: Box<[ArrayId]>,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::TensorCopyKernel::new(context, data_type)?;
        Ok(Self {
            kernel,
            argument_arrays,
        })
    }
}

impl<B: Backend> EncodableBlock<B> for TensorCopy<B> {
    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
    ) {
        let arrays = state.arrays(&self.argument_arrays);
        assert_eq!(arrays.len(), 2, "TensorCopy expects exactly 2 arrays");

        let length = arrays[0].borrow().num_elements();

        let source_array = arrays[0].borrow_mut();
        let destination_array = arrays[1].borrow_mut();

        self.kernel.encode(source_array.buffer(), destination_array.buffer(), length as u32, encoder);
    }
}
