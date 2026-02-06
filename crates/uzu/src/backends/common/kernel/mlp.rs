use crate::{
    DataType,
    backends::common::{
        Backend, Context, Kernels,
        kernel::MlpGateActMulKernel,
    },
    config::Activation,
};

/// MLP activation type for fused kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum MlpActivationType {
    SiLU = 0,
    Gelu = 1,
}

impl From<&Activation> for MlpActivationType {
    fn from(act: &Activation) -> Self {
        match act {
            Activation::SiLU {
                ..
            } => MlpActivationType::SiLU,
            Activation::Gelu => MlpActivationType::Gelu,
            Activation::Identity => {
                panic!("Identity activation not supported for MLP fusion")
            },
        }
    }
}

pub struct MlpGateKernel<B: Backend> {
    kernel: <B::Kernels as Kernels>::MlpGateActMulKernel,
    activation: Activation,
    hidden_dim: usize,
}

impl<B: Backend> MlpGateKernel<B> {
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
        encoder: &B::EncoderRef,
        fused_up: &B::NativeBuffer,
        hidden: &B::NativeBuffer,
        m: i32,
    ) -> Result<(), B::Error> {
        let act_type = match self.activation {
            Activation::SiLU { .. } => 0,
            Activation::Gelu => 1,
            Activation::Identity => panic!("Identity activation is not supported for kernel"),
        };
        
        self.kernel.encode(
            fused_up,
            hidden,
            self.hidden_dim as i32,
            m,
            act_type,
            encoder,
        );
        Ok(())
    }
}
