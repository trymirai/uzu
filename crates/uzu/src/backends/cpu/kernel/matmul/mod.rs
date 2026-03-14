use crate::{
    DataType,
    backends::{
        common::kernel::matmul::{
            MatmulArguments, MatmulKernel as MatmulKernelTrait, MatmulError,
            MatmulKernels,
        },
        cpu::{Cpu, command_buffer::CpuCommandBuffer, context::CpuContext, kernel::CpuKernels},
    },
};

pub mod gemm;
pub mod gemm_mpp;
pub mod gemv;

pub struct MatmulCpuKernel;

impl MatmulKernelTrait for MatmulCpuKernel {
    type Backend = Cpu;

    fn new(
        #[allow(unused)] context: &CpuContext,
        #[allow(unused)] data_type: DataType,
    ) -> Result<Self, MatmulError<Cpu>> {
        Ok(Self)
    }

    fn encode(
        &mut self,
        #[allow(unused)] context: &CpuContext,
        #[allow(unused)] encoder: &mut CpuCommandBuffer,
        #[allow(unused)] arguments: MatmulArguments<Cpu>,
    ) {
        encoder.push_command(move || todo!());
    }
}

impl MatmulKernels for CpuKernels {
    type MatmulKernel = MatmulCpuKernel;
}
