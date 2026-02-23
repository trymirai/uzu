use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            gpu_types::{GEMMParams, GEMMSpiltKParams},
            kernel::{
                BufferArg, MatmulGemmKernel, MatmulGemvKernel, MatmulSplitKAccumBfloat16Kernel,
                MatmulSplitKPartialBfloat16Kernel,
            },
        },
        cpu::backend::Cpu,
    },
};

pub struct MatmulGemmCpuKernel;

impl MatmulGemmKernel for MatmulGemmCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _block_rows: u32,
        _block_cols: u32,
        _block_depth: u32,
        _warps_per_row: u32,
        _warps_per_col: u32,
        _align_m: bool,
        _align_n: bool,
        _align_k: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'a, 'b, 'd, 'encoder>(
        &self,
        _a: impl BufferArg<'a, <Self::Backend as Backend>::NativeBuffer>,
        _b: impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>,
        _d: impl BufferArg<'d, <Self::Backend as Backend>::NativeBuffer>,
        _params: &[GEMMParams],
        _group_count_x: u32,
        _group_count_y: u32,
        _group_count_z: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'a, 'b, 'd, 'encoder, 'predicate>(
        &self,
        _a: impl BufferArg<'a, <Self::Backend as Backend>::NativeBuffer>,
        _b: impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>,
        _d: impl BufferArg<'d, <Self::Backend as Backend>::NativeBuffer>,
        _params: &[GEMMParams],
        _group_count_x: u32,
        _group_count_y: u32,
        _group_count_z: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MatmulGemvCpuKernel;

impl MatmulGemvKernel for MatmulGemvCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _tg_simd_rows: u32,
        _tg_simd_cols: u32,
        _sg_thread_rows: u32,
        _sg_thread_cols: u32,
        _thread_out_rows: u32,
        _thread_out_cols: u32,
        _apply_output_scale_and_accumulate: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'matrix, 'input_vector, 'output_source, 'output_vector, 'encoder>(
        &self,
        _matrix: impl BufferArg<'matrix, <Self::Backend as Backend>::NativeBuffer>,
        _input_vector: impl BufferArg<'input_vector, <Self::Backend as Backend>::NativeBuffer>,
        _output_source: Option<impl BufferArg<'output_source, <Self::Backend as Backend>::NativeBuffer>>,
        _output_vector: impl BufferArg<'output_vector, <Self::Backend as Backend>::NativeBuffer>,
        _input_dimension: i32,
        _output_dimension: i32,
        _matrix_leading_dimension: i32,
        _output_scale: f32,
        _output_accumulate_scale: f32,
        _batch_shape: &[i32],
        _vector_batch_stride: &[i32],
        _matrix_batch_stride: &[i32],
        _output_source_batch_stride: &[i32],
        _output_source_stride: i32,
        _batch_rows: i32,
        _output_rows_per_threadgroup: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'matrix, 'input_vector, 'output_source, 'output_vector, 'encoder, 'predicate>(
        &self,
        _matrix: impl BufferArg<'matrix, <Self::Backend as Backend>::NativeBuffer>,
        _input_vector: impl BufferArg<'input_vector, <Self::Backend as Backend>::NativeBuffer>,
        _output_source: Option<impl BufferArg<'output_source, <Self::Backend as Backend>::NativeBuffer>>,
        _output_vector: impl BufferArg<'output_vector, <Self::Backend as Backend>::NativeBuffer>,
        _input_dimension: i32,
        _output_dimension: i32,
        _matrix_leading_dimension: i32,
        _output_scale: f32,
        _output_accumulate_scale: f32,
        _batch_shape: &[i32],
        _vector_batch_stride: &[i32],
        _matrix_batch_stride: &[i32],
        _output_source_batch_stride: &[i32],
        _output_source_stride: i32,
        _batch_rows: i32,
        _output_rows_per_threadgroup: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

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
