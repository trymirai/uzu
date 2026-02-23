use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MatmulSplitKAccumBfloat16Kernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MatmulSplitKAccumBfloat16CpuKernel;

impl MatmulSplitKAccumBfloat16Kernel for MatmulSplitKAccumBfloat16CpuKernel {
    type Backend = Cpu;

    fn new(_context: &<Self::Backend as Backend>::Context) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'c_split, 'd, 'encoder>(
        &self,
        _c_split: impl BufferArg<'c_split, <Self::Backend as Backend>::NativeBuffer>,
        _d: impl BufferArg<'d, <Self::Backend as Backend>::NativeBuffer>,
        _k_partitions: i32,
        _partition_stride: i32,
        _ldd: i32,
        _accum_total_threads_x: u32,
        _accum_total_threads_y: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'c_split, 'd, 'encoder, 'predicate>(
        &self,
        _c_split: impl BufferArg<'c_split, <Self::Backend as Backend>::NativeBuffer>,
        _d: impl BufferArg<'d, <Self::Backend as Backend>::NativeBuffer>,
        _k_partitions: i32,
        _partition_stride: i32,
        _ldd: i32,
        _accum_total_threads_x: u32,
        _accum_total_threads_y: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
