use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeBuildTileMapKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeBuildTileMapCpuKernel;

impl MoeBuildTileMapKernel for MoeBuildTileMapCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'offsets, 'tile_row_offsets, 'tile_counts, 'tile_map, 'encoder>(
        &self,
        _offsets: impl BufferArg<'offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_row_offsets: impl BufferArg<
            'tile_row_offsets,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _tile_map: impl BufferArg<'tile_map, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'offsets, 'tile_row_offsets, 'tile_counts, 'tile_map, 'encoder, 'predicate>(
        &self,
        _offsets: impl BufferArg<'offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_row_offsets: impl BufferArg<
            'tile_row_offsets,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _tile_map: impl BufferArg<'tile_map, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
