use crate::{
    DataType,
    backends::common::{Backend, CommandBuffer, Kernels, gpu_types::ActivationType, kernel::MlpGateActMulKernel},
};

pub struct MlpGateActMulEncodable<B: Backend> {
    kernel: <B::Kernels as Kernels>::MlpGateActMulKernel,
    activation: ActivationType,
    hidden_dim: usize,
}

impl<B: Backend> MlpGateActMulEncodable<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        activation: ActivationType,
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
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        fused_up: &B::Buffer,
        hidden: &mut B::Buffer,
        m: i32,
    ) -> Result<(), B::Error> {
        if self.activation == ActivationType::IDENTITY {
            panic!("Identity activation is not supported for kernel")
        }
        self.kernel.encode(fused_up, hidden, self.hidden_dim as i32, m, self.activation, command_buffer);
        Ok(())
    }
}
