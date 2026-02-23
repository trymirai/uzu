use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, AudioCausalConv1dKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct AudioCausalConv1dCpuKernel;

impl AudioCausalConv1dKernel for AudioCausalConv1dCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'weight, 'bias, 'output, 'lengths, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _weight: impl BufferArg<'weight, <Self::Backend as Backend>::NativeBuffer>,
        _bias: impl BufferArg<'bias, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _lengths: impl BufferArg<'lengths, <Self::Backend as Backend>::NativeBuffer>,
        _cin: i32,
        _cout: i32,
        _seq_len: i32,
        _kernel_size: i32,
        _dilation: i32,
        _batch_size: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'weight, 'bias, 'output, 'lengths, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _weight: impl BufferArg<'weight, <Self::Backend as Backend>::NativeBuffer>,
        _bias: impl BufferArg<'bias, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _lengths: impl BufferArg<'lengths, <Self::Backend as Backend>::NativeBuffer>,
        _cin: i32,
        _cout: i32,
        _seq_len: i32,
        _kernel_size: i32,
        _dilation: i32,
        _batch_size: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
