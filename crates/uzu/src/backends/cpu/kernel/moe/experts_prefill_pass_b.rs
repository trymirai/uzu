use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeExpertsPrefillPassBKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeExpertsPrefillPassBCpuKernel;

impl MoeExpertsPrefillPassBKernel for MoeExpertsPrefillPassBCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<
        'hidden,
        'expert_offsets,
        'w2_all,
        'down_biases,
        'output,
        'tile_map,
        '__dsl_indirect_dispatch_buffer,
        'encoder,
    >(
        &self,
        _hidden: impl BufferArg<'hidden, <Self::Backend as Backend>::NativeBuffer>,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _w2_all: impl BufferArg<'w2_all, <Self::Backend as Backend>::NativeBuffer>,
        _down_biases: impl BufferArg<'down_biases, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _d_ff: u32,
        _e: u32,
        _tile_map: impl BufferArg<'tile_map, <Self::Backend as Backend>::NativeBuffer>,
        __dsl_indirect_dispatch_buffer: impl BufferArg<
            '__dsl_indirect_dispatch_buffer,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<
        'hidden,
        'expert_offsets,
        'w2_all,
        'down_biases,
        'output,
        'tile_map,
        '__dsl_indirect_dispatch_buffer,
        'encoder,
        'predicate,
    >(
        &self,
        _hidden: impl BufferArg<'hidden, <Self::Backend as Backend>::NativeBuffer>,
        _expert_offsets: impl BufferArg<'expert_offsets, <Self::Backend as Backend>::NativeBuffer>,
        _w2_all: impl BufferArg<'w2_all, <Self::Backend as Backend>::NativeBuffer>,
        _down_biases: impl BufferArg<'down_biases, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _d_model: u32,
        _d_ff: u32,
        _e: u32,
        _tile_map: impl BufferArg<'tile_map, <Self::Backend as Backend>::NativeBuffer>,
        __dsl_indirect_dispatch_buffer: impl BufferArg<
            '__dsl_indirect_dispatch_buffer,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
