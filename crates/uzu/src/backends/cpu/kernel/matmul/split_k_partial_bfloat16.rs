use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            gpu_types::GEMMSpiltKParams,
            kernel::{BufferArg, MatmulSplitKPartialBfloat16Kernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MatmulSplitKPartialBfloat16CpuKernel;

impl MatmulSplitKPartialBfloat16Kernel for MatmulSplitKPartialBfloat16CpuKernel {
    type Backend = Cpu;

    fn new(_context: &<Self::Backend as Backend>::Context) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'a, 'b, 'c, 'encoder>(
        &self,
        _a: impl BufferArg<'a, <Self::Backend as Backend>::NativeBuffer>,
        _b: impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>,
        _c: impl BufferArg<'c, <Self::Backend as Backend>::NativeBuffer>,
        _params: &[GEMMSpiltKParams],
        _partial_group_count_x: u32,
        _partial_group_count_y: u32,
        _partial_group_count_z: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'a, 'b, 'c, 'encoder, 'predicate>(
        &self,
        _a: impl BufferArg<'a, <Self::Backend as Backend>::NativeBuffer>,
        _b: impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>,
        _c: impl BufferArg<'c, <Self::Backend as Backend>::NativeBuffer>,
        _params: &[GEMMSpiltKParams],
        _partial_group_count_x: u32,
        _partial_group_count_y: u32,
        _partial_group_count_z: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
