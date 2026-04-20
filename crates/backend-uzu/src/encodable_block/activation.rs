//! Activation encodable.

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        ActivationConfig, Allocation, Backend, Encoder,
        gpu_types::ActivationType,
        kernel::{ActivationKernel, Kernels},
    },
};

pub struct Activation<B: Backend> {
    kernel: <B::Kernels as Kernels>::ActivationKernel,
    activation: ActivationConfig,
    data_type: DataType,
}

impl<B: Backend> Activation<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        config: ActivationConfig,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::ActivationKernel::new(context, data_type, false)?;
        Ok(Self {
            kernel,
            activation: config,
            data_type,
        })
    }

    pub fn encode(
        &self,
        input: &Allocation<B>,
        n: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        if self.activation.act_type() == ActivationType::IDENTITY {
            panic!("Identity activation is not supported for kernel");
        }

        let mut output = encoder.allocate_scratch(size_for_shape(&[n], self.data_type))?;
        self.kernel.encode(Some(input), &mut output, n as u32, self.activation.act_type(), encoder);
        Ok(output)
    }
}
