//! Tensor add-swap encodable.

use std::ops::DerefMut;

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder,
        kernel::{Kernels, TensorAddSwapKernel},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub struct TensorAddSwap<B: Backend> {
    kernel: <B::Kernels as Kernels>::TensorAddSwapKernel,
    skip_array_id: ArrayId,
    main_array_id: ArrayId,
}

impl<B: Backend> TensorAddSwap<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        skip_array_id: ArrayId,
        main_array_id: ArrayId,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::TensorAddSwapKernel::new(context, data_type)?;

        Ok(Self {
            kernel,
            skip_array_id,
            main_array_id,
        })
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let skip_array = state.array(self.skip_array_id);
        let main_array = state.array(self.main_array_id);

        let length = skip_array.num_elements();

        self.kernel.encode(
            skip_array.buffer().borrow_mut().deref_mut(),
            main_array.buffer().borrow_mut().deref_mut(),
            length as u32,
            encoder,
        );
        Ok(())
    }
}
