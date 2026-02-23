use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoePassATileScanKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoePassATileScanCpuKernel;

impl MoePassATileScanKernel for MoePassATileScanCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'tile_counts, 'tile_offsets, 'total_tiles, 'encoder>(
        &self,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _tile_offsets: impl BufferArg<'tile_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _total_tiles: impl BufferArg<'total_tiles, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'tile_counts, 'tile_offsets, 'total_tiles, 'encoder, 'predicate>(
        &self,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _tile_offsets: impl BufferArg<'tile_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _total_tiles: impl BufferArg<'total_tiles, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
