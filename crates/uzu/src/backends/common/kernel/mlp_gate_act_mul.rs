use crate::{
    DataType,
    backends::common::{Backend, Kernels, kernel::MlpGateActMulKernel},
    config::Activation,
};

pub struct MlpGateActMulEncodable<B: Backend> {
    kernel: <B::Kernels as Kernels>::MlpGateActMulKernel,
    activation: Activation,
    hidden_dim: usize,
}

impl<B: Backend> MlpGateActMulEncodable<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        activation: Activation,
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
        encoder: &B::ComputeEncoder,
        fused_up: &B::NativeBuffer,
        hidden: &B::NativeBuffer,
        m: i32,
    ) -> Result<(), B::Error> {
        let act_type = match self.activation {
            Activation::SiLU {
                ..
            } => 0,
            Activation::Gelu => 1,
            Activation::Identity => {
                panic!("Identity activation is not supported for kernel")
            },
        };
        self.kernel.encode(fused_up, hidden, self.hidden_dim as i32, m, act_type, encoder);
        Ok(())
    }
}
