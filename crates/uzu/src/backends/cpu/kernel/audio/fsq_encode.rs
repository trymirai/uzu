use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, AudioFsqEncodeKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct AudioFsqEncodeCpuKernel;

impl AudioFsqEncodeKernel for AudioFsqEncodeCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'tokens, 'lengths, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _tokens: impl BufferArg<'tokens, <Self::Backend as Backend>::NativeBuffer>,
        _lengths: impl BufferArg<'lengths, <Self::Backend as Backend>::NativeBuffer>,
        _num_groups: i32,
        _seq_len: i32,
        _codebook_dim: i32,
        _num_levels: &[i32],
        _dim_base_index: &[i32],
        _eps: f32,
        _batch_size: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'tokens, 'lengths, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _tokens: impl BufferArg<'tokens, <Self::Backend as Backend>::NativeBuffer>,
        _lengths: impl BufferArg<'lengths, <Self::Backend as Backend>::NativeBuffer>,
        _num_groups: i32,
        _seq_len: i32,
        _codebook_dim: i32,
        _num_levels: &[i32],
        _dim_base_index: &[i32],
        _eps: f32,
        _batch_size: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
