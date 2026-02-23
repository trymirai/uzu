use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{AttentionTwoPass2Kernel, BufferArg},
        },
        cpu::Cpu,
    },
};

pub struct AttentionTwoPass2CpuKernel;

impl AttentionTwoPass2Kernel for AttentionTwoPass2CpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _head_dim: u32,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'partials, 'sums, 'maxs, 'out, 'encoder>(
        &self,
        _partials: impl BufferArg<'partials, <Self::Backend as Backend>::NativeBuffer>,
        _sums: impl BufferArg<'sums, <Self::Backend as Backend>::NativeBuffer>,
        _maxs: impl BufferArg<'maxs, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _num_heads: u32,
        _suffix_length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'partials, 'sums, 'maxs, 'out, 'encoder, 'predicate>(
        &self,
        _partials: impl BufferArg<'partials, <Self::Backend as Backend>::NativeBuffer>,
        _sums: impl BufferArg<'sums, <Self::Backend as Backend>::NativeBuffer>,
        _maxs: impl BufferArg<'maxs, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _num_heads: u32,
        _suffix_length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
