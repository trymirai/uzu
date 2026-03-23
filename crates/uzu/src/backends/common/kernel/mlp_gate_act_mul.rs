use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{
    DataType,
    backends::common::{
        ActivationConfig, Backend, CommandBuffer, Kernels,
        gpu_types::ActivationType,
        kernel::{MlpGateActMulHadamardKernel, MlpGateActMulKernel},
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
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        fused_up: &B::Buffer,
        hidden: &mut B::Buffer,
        m: i32,
    ) -> Result<(), B::Error> {
        if self.activation.act_type() == ActivationType::IDENTITY {
            panic!("Identity activation is not supported for kernel")
        }
        self.kernel.encode(fused_up, hidden, self.hidden_dim as i32, m, self.activation.act_type(), command_buffer);
        Ok(())
    }
}

pub struct MlpGateActMulHadamardEncodable<B: Backend> {
    kernel: <B::Kernels as Kernels>::MlpGateActMulHadamardKernel,
    activation: ActivationConfig,
    hidden_dim: usize,
    hadamard_factors: Rc<RefCell<B::Buffer>>,
}

impl<B: Backend> MlpGateActMulHadamardEncodable<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        activation: ActivationConfig,
        hidden_dim: usize,
        hadamard_factors: Rc<RefCell<B::Buffer>>,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::MlpGateActMulHadamardKernel::new(context, data_type)?;
        Ok(Self {
            kernel,
            activation,
            hidden_dim,
            hadamard_factors,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        fused_up: &B::Buffer,
        hidden: &mut B::Buffer,
        m: i32,
    ) -> Result<(), B::Error> {
        if self.activation.act_type() == ActivationType::IDENTITY {
            panic!("Identity activation is not supported for kernel")
        }
        self.kernel.encode(
            fused_up,
            hidden,
            self.hadamard_factors.borrow().deref(),
            self.hidden_dim as i32,
            m,
            self.activation.act_type(),
            command_buffer,
        );
        Ok(())
    }
}
