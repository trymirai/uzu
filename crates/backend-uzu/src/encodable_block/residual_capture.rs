use crate::{
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, TensorAddScaleKernel},
    },
    data_type::DataType,
};

pub(super) struct ResidualCapture<B: Backend> {
    model_dim: usize,
    add: <B::Kernels as Kernels>::TensorAddScaleKernel,
}

impl<B: Backend> ResidualCapture<B> {
    pub(super) fn new(
        context: &B::Context,
        model_dim: usize,
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            model_dim,
            add: <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, false)?,
        })
    }

    pub(super) fn encode(
        &self,
        shortcut: &Allocation<B>,
        hidden: &Allocation<B>,
        batch_size: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut output = encoder.allocate_scratch(hidden.size())?;
        let elements = (batch_size * self.model_dim) as u32;
        self.add.encode(Some(shortcut), hidden, &mut output, elements, elements, 1.0, encoder);
        Ok(output)
    }
}
