use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeTileCountsKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeTileCountsCpuKernel;

impl MoeTileCountsKernel for MoeTileCountsCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'offsets, 'tile_counts, 'encoder>(
        &self,
        _offsets: impl BufferArg<'offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'offsets, 'tile_counts, 'encoder, 'predicate>(
        &self,
        _offsets: impl BufferArg<'offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
