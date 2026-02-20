use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{
                BufferArg, TensorAddBiasKernel, TensorAddSwapKernel, TensorCopyKernel,
                TokenCopySampledKernel, TokenCopyToResultsKernel,
            },
        },
        cpu::backend::Cpu,
    },
};

pub struct TensorAddBiasCpuKernel;

impl TensorAddBiasKernel for TensorAddBiasCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'bias, 'output, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _bias: impl BufferArg<'bias, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _num_cols: u32,
        _length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'bias, 'output, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _bias: impl BufferArg<'bias, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _num_cols: u32,
        _length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct TensorAddSwapCpuKernel;

impl TensorAddSwapKernel for TensorAddSwapCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'skip_buffer, 'main_buffer, 'encoder>(
        &self,
        _skip_buffer: impl BufferArg<'skip_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _main_buffer: impl BufferArg<'main_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'skip_buffer, 'main_buffer, 'encoder, 'predicate>(
        &self,
        _skip_buffer: impl BufferArg<'skip_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _main_buffer: impl BufferArg<'main_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct TensorCopyCpuKernel;

impl TensorCopyKernel for TensorCopyCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'src_buffer, 'dst_buffer, 'encoder>(
        &self,
        _src_buffer: impl BufferArg<'src_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _dst_buffer: impl BufferArg<'dst_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'src_buffer, 'dst_buffer, 'encoder, 'predicate>(
        &self,
        _src_buffer: impl BufferArg<'src_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _dst_buffer: impl BufferArg<'dst_buffer, <Self::Backend as Backend>::NativeBuffer>,
        _length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct TokenCopySampledCpuKernel;

impl TokenCopySampledKernel for TokenCopySampledCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'src, 'dst, 'encoder>(
        &self,
        _src: impl BufferArg<'src, <Self::Backend as Backend>::NativeBuffer>,
        _dst: impl BufferArg<'dst, <Self::Backend as Backend>::NativeBuffer>,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'src, 'dst, 'encoder, 'predicate>(
        &self,
        _src: impl BufferArg<'src, <Self::Backend as Backend>::NativeBuffer>,
        _dst: impl BufferArg<'dst, <Self::Backend as Backend>::NativeBuffer>,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct TokenCopyToResultsCpuKernel;

impl TokenCopyToResultsKernel for TokenCopyToResultsCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'src, 'dst, 'encoder>(
        &self,
        _src: impl BufferArg<'src, <Self::Backend as Backend>::NativeBuffer>,
        _dst: impl BufferArg<'dst, <Self::Backend as Backend>::NativeBuffer>,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'src, 'dst, 'encoder, 'predicate>(
        &self,
        _src: impl BufferArg<'src, <Self::Backend as Backend>::NativeBuffer>,
        _dst: impl BufferArg<'dst, <Self::Backend as Backend>::NativeBuffer>,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
