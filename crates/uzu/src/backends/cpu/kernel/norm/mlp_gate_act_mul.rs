use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MlpGateActMulKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MlpGateActMulCpuKernel;

impl MlpGateActMulKernel for MlpGateActMulCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'fused_up, 'hidden, 'encoder>(
        &self,
        _fused_up: impl BufferArg<'fused_up, <Self::Backend as Backend>::NativeBuffer>,
        _hidden: impl BufferArg<'hidden, <Self::Backend as Backend>::NativeBuffer>,
        _h: i32,
        _m: i32,
        _act_type: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'fused_up, 'hidden, 'encoder, 'predicate>(
        &self,
        _fused_up: impl BufferArg<'fused_up, <Self::Backend as Backend>::NativeBuffer>,
        _hidden: impl BufferArg<'hidden, <Self::Backend as Backend>::NativeBuffer>,
        _h: i32,
        _m: i32,
        _act_type: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
