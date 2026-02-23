use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeTileScanKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeTileScanCpuKernel;

impl MoeTileScanKernel for MoeTileScanCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'tile_counts, 'tile_row_offsets, 'total_tiles_buf, 'encoder>(
        &self,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _tile_row_offsets: impl BufferArg<
            'tile_row_offsets,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _total_tiles_buf: impl BufferArg<'total_tiles_buf, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'tile_counts, 'tile_row_offsets, 'total_tiles_buf, 'encoder, 'predicate>(
        &self,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _tile_row_offsets: impl BufferArg<
            'tile_row_offsets,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _total_tiles_buf: impl BufferArg<'total_tiles_buf, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
