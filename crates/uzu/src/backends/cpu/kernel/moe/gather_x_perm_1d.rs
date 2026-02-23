use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeGatherXPerm1DKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeGatherXPerm1DCpuKernel;

impl MoeGatherXPerm1DKernel for MoeGatherXPerm1DCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'x, 'bucketed_ids, 'x_perm, 'sumk_buf, 'encoder>(
        &self,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _bucketed_ids: impl BufferArg<'bucketed_ids, <Self::Backend as Backend>::NativeBuffer>,
        _x_perm: impl BufferArg<'x_perm, <Self::Backend as Backend>::NativeBuffer>,
        _sumk_buf: impl BufferArg<'sumk_buf, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _t: u32,
        _k: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'x, 'bucketed_ids, 'x_perm, 'sumk_buf, 'encoder, 'predicate>(
        &self,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _bucketed_ids: impl BufferArg<'bucketed_ids, <Self::Backend as Backend>::NativeBuffer>,
        _x_perm: impl BufferArg<'x_perm, <Self::Backend as Backend>::NativeBuffer>,
        _sumk_buf: impl BufferArg<'sumk_buf, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _t: u32,
        _k: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
