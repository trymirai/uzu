use std::ops::DerefMut;

use crate::{
    DataType,
    backends::common::{Backend, Encoder, Kernels, kernel::ValueNormKernel},
    encodable_block::model_extension::ModelExtensionError,
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub struct ValueNorm<B: Backend> {
    kernel: <B::Kernels as Kernels>::ValueNormKernel,
    num_heads: usize,
    num_groups: usize,
    head_dim: usize,
}

impl<B: Backend> ValueNorm<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        num_heads: usize,
        num_groups: usize,
        head_dim: usize,
    ) -> Result<Self, ModelExtensionError<B>> {
        let kernel = <B::Kernels as Kernels>::ValueNormKernel::new(context, data_type)
            .map_err(ModelExtensionError::BackendError)?;
        Ok(Self {
            kernel,
            num_heads,
            num_groups,
            head_dim,
        })
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) {
        let qkv = state.array(ArrayId::QKV);
        self.kernel.encode(
            qkv.buffer().borrow_mut().deref_mut(),
            state.active_row_count() as u32,
            self.num_heads as u32,
            self.num_groups as u32,
            self.head_dim as u32,
            1e-6,
            encoder,
        );
    }
}
