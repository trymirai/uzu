use crate::{
    DataType,
    backends::common::{
        ActivationConfig, Allocation, Backend, Encoder, Kernels, gpu_types::ActivationType, kernel::MlpGateActMulKernel,
    },
};

pub struct MlpGateActMulEncodable<B: Backend> {
    kernel: <B::Kernels as Kernels>::MlpGateActMulKernel,
    activation: ActivationConfig,
    hidden_dim: usize,
}

impl<B: Backend> MlpGateActMulEncodable<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        activation: ActivationConfig,
        hidden_dim: usize,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::MlpGateActMulKernel::new(context, data_type)?;
        Ok(Self {
            kernel,
            activation,
            hidden_dim,
        })
    }

    pub fn encode(
        &self,
        encoder: &mut Encoder<B>,
        fused_up: &Allocation<B>,
        hidden: &mut Allocation<B>,
        m: i32,
    ) -> Result<(), B::Error> {
        if self.activation.act_type() == ActivationType::IDENTITY {
            panic!("Identity activation is not supported for kernel")
        }
        self.kernel.encode(fused_up, hidden, self.hidden_dim as i32, m, self.activation.act_type(), encoder);
        Ok(())
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}
