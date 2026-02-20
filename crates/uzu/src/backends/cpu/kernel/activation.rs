use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{ActivationKernel, BufferArg},
        },
        cpu::backend::Cpu,
    },
};

pub struct ActivationCpuKernel;

impl ActivationKernel for ActivationCpuKernel {
    type Backend = Cpu;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'output, 'encoder>(
        &self,
        input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        n: u32,
        act_type: u32,
        encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'output, 'encoder, 'predicate>(
        &self,
        input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        n: u32,
        act_type: u32,
        encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
