use crate::{
    array::size_for_shape,
    backends::common::{Allocation, Backend, Encoder, Kernels, gpu_types::ActivationType, kernel::GatedActMulKernel},
    config::activation::AnyActivation,
    data_type::DataType,
};

pub struct MlpGateActMulEncodable<B: Backend> {
    kernel: <B::Kernels as Kernels>::GatedActMulKernel,
    activation: AnyActivation,
    hidden_dim: usize,
    data_type: DataType,
    hadamard_factors: Option<Allocation<B>>,
}

impl<B: Backend> MlpGateActMulEncodable<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        activation: AnyActivation,
        hidden_dim: usize,
        hadamard_factors: Option<Allocation<B>>,
    ) -> Result<Self, B::Error> {
        let kernel =
            <B::Kernels as Kernels>::GatedActMulKernel::new(context, data_type, true, hadamard_factors.is_some())?;
        Ok(Self {
            kernel,
            activation,
            hidden_dim,
            data_type,
            hadamard_factors,
        })
    }

    pub fn encode(
        &self,
        encoder: &mut Encoder<B>,
        fused_up: &Allocation<B>,
        batch_dim: usize,
    ) -> Result<Allocation<B>, B::Error> {
        if self.activation.act_type() == ActivationType::IDENTITY {
            panic!("Identity activation is not supported for kernel")
        }
        let mut hidden = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.hidden_dim], self.data_type))?;
        self.kernel.encode(
            fused_up,
            None::<&Allocation<B>>,
            &mut hidden,
            self.hadamard_factors.as_ref(),
            self.hidden_dim as u32,
            batch_dim as u32,
            0,
            0,
            self.activation.act_type(),
            encoder,
        );
        Ok(hidden)
    }
}
