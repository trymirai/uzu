use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoePassABuildTileMapKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoePassABuildTileMapCpuKernel;

impl MoePassABuildTileMapKernel for MoePassABuildTileMapCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'expert_offsets, 'tile_offsets, 'row_expert_map, 'tile_map, 'encoder>(
        &self,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_offsets: impl BufferArg<'tile_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _row_expert_map: impl BufferArg<'row_expert_map, <Self::Backend as Backend>::NativeBuffer>,
        _tile_map: impl BufferArg<'tile_map, <Self::Backend as Backend>::NativeBuffer>,
        _total_rows: u32,
        _h_blocks: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<
        'expert_offsets,
        'tile_offsets,
        'row_expert_map,
        'tile_map,
        'encoder,
        'predicate,
    >(
        &self,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_offsets: impl BufferArg<'tile_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _row_expert_map: impl BufferArg<'row_expert_map, <Self::Backend as Backend>::NativeBuffer>,
        _tile_map: impl BufferArg<'tile_map, <Self::Backend as Backend>::NativeBuffer>,
        _total_rows: u32,
        _h_blocks: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
