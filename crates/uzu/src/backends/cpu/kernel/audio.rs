use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{
                AudioAddKernel, AudioCausalConv1dKernel, AudioCausalConvTranspose1dKernel,
                AudioClampKernel, AudioConv1dKernel, AudioFsqDecodeKernel, AudioFsqEncodeKernel,
                AudioHalfSnakeKernel, AudioLeakyReluKernel, AudioScaleKernel, AudioTanhKernel,
                BufferArg,
            },
        },
        cpu::backend::Cpu,
    },
};

pub struct AudioFsqDecodeCpuKernel;

impl AudioFsqDecodeKernel for AudioFsqDecodeCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'tokens, 'out, 'lengths, 'encoder>(
        &self,
        _tokens: impl BufferArg<'tokens, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _lengths: impl BufferArg<'lengths, <Self::Backend as Backend>::NativeBuffer>,
        _num_groups: i32,
        _seq_len: i32,
        _codebook_dim: i32,
        _num_levels: &[i32],
        _batch_size: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'tokens, 'out, 'lengths, 'encoder, 'predicate>(
        &self,
        _tokens: impl BufferArg<'tokens, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _lengths: impl BufferArg<'lengths, <Self::Backend as Backend>::NativeBuffer>,
        _num_groups: i32,
        _seq_len: i32,
        _codebook_dim: i32,
        _num_levels: &[i32],
        _batch_size: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct AudioLeakyReluCpuKernel;

impl AudioLeakyReluKernel for AudioLeakyReluCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'output, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _n: i32,
        _negative_slope: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'output, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _n: i32,
        _negative_slope: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct AudioTanhCpuKernel;

impl AudioTanhKernel for AudioTanhCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'output, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _n: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'output, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _n: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct AudioAddCpuKernel;

impl AudioAddKernel for AudioAddCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'a, 'b, 'out, 'encoder>(
        &self,
        _a: impl BufferArg<'a, <Self::Backend as Backend>::NativeBuffer>,
        _b: impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _n: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'a, 'b, 'out, 'encoder, 'predicate>(
        &self,
        _a: impl BufferArg<'a, <Self::Backend as Backend>::NativeBuffer>,
        _b: impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _n: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct AudioScaleCpuKernel;

impl AudioScaleKernel for AudioScaleCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'output, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _n: i32,
        _scale: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'output, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _n: i32,
        _scale: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

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

pub struct AudioCausalConvTranspose1dCpuKernel;

impl AudioCausalConvTranspose1dKernel for AudioCausalConvTranspose1dCpuKernel {
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
        _seq_len_in: i32,
        _seq_len_out: i32,
        _stride: i32,
        _groups: i32,
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
        _seq_len_in: i32,
        _seq_len_out: i32,
        _stride: i32,
        _groups: i32,
        _batch_size: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct AudioHalfSnakeCpuKernel;

impl AudioHalfSnakeKernel for AudioHalfSnakeCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'alpha, 'output, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _alpha: impl BufferArg<'alpha, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _channels: i32,
        _seq_len: i32,
        _snake_channels: i32,
        _negative_slope: f32,
        _eps: f32,
        _batch_size: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'alpha, 'output, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _alpha: impl BufferArg<'alpha, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _channels: i32,
        _seq_len: i32,
        _snake_channels: i32,
        _negative_slope: f32,
        _eps: f32,
        _batch_size: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct AudioClampCpuKernel;

impl AudioClampKernel for AudioClampCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'output, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _n: i32,
        _min_value: f32,
        _max_value: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'output, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _n: i32,
        _min_value: f32,
        _max_value: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct AudioConv1dCpuKernel;

impl AudioConv1dKernel for AudioConv1dCpuKernel {
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
        _seq_len_in: i32,
        _seq_len_out: i32,
        _kernel_size: i32,
        _stride: i32,
        _dilation: i32,
        _padding: i32,
        _pad_mode: i32,
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
        _seq_len_in: i32,
        _seq_len_out: i32,
        _kernel_size: i32,
        _stride: i32,
        _dilation: i32,
        _padding: i32,
        _pad_mode: i32,
        _batch_size: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

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
