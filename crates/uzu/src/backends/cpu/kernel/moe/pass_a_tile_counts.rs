use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoePassATileCountsKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoePassATileCountsCpuKernel;

impl MoePassATileCountsKernel for MoePassATileCountsCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'expert_offsets, 'tile_counts, 'encoder>(
        &self,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _h_blocks: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'expert_offsets, 'tile_counts, 'encoder, 'predicate>(
        &self,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _tile_counts: impl BufferArg<'tile_counts, <Self::Backend as Backend>::NativeBuffer>,
        _e: u32,
        _h_blocks: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
