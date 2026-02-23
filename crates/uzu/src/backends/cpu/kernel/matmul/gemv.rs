use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MatmulGemvKernel},
        },
        cpu::backend::Cpu,
    },
};

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
