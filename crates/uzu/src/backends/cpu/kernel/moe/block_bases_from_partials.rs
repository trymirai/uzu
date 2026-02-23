use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeBlockBasesFromPartialsKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeBlockBasesFromPartialsCpuKernel;

impl MoeBlockBasesFromPartialsKernel for MoeBlockBasesFromPartialsCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'partials, 'block_bases, 'block_alloc, 'encoder>(
        &self,
        _partials: impl BufferArg<'partials, <Self::Backend as Backend>::NativeBuffer>,
        _block_bases: impl BufferArg<'block_bases, <Self::Backend as Backend>::NativeBuffer>,
        _block_alloc: impl BufferArg<'block_alloc, <Self::Backend as Backend>::NativeBuffer>,
        _e_input: u32,
        _num_blocks: u32,
        _num_tiles: u32,
        _capacity_per_expert: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'partials, 'block_bases, 'block_alloc, 'encoder, 'predicate>(
        &self,
        _partials: impl BufferArg<'partials, <Self::Backend as Backend>::NativeBuffer>,
        _block_bases: impl BufferArg<'block_bases, <Self::Backend as Backend>::NativeBuffer>,
        _block_alloc: impl BufferArg<'block_alloc, <Self::Backend as Backend>::NativeBuffer>,
        _e_input: u32,
        _num_blocks: u32,
        _num_tiles: u32,
        _capacity_per_expert: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
