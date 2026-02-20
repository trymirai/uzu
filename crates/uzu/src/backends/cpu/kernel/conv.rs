use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{
                BufferArg, Conv1dDecodeKernel, Conv1dPackKernel, Conv1dScanKernel,
                ShortConvDecodeKernel, ShortConvPackKernel, ShortConvPrefillKernel,
                ShortConvTrieKernel, SigmoidKernel, SplitInProjKernel,
            },
        },
        cpu::backend::Cpu,
    },
};

pub struct ShortConvPackCpuKernel;

impl ShortConvPackKernel for ShortConvPackCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'state_in, 'in_proj, 'padded, 'encoder>(
        &self,
        _state_in: impl BufferArg<'state_in, <Self::Backend as Backend>::NativeBuffer>,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _state_stride: u32,
        _suffix_len: u32,
        _in_proj_stride: u32,
        _model_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'state_in, 'in_proj, 'padded, 'encoder, 'predicate>(
        &self,
        _state_in: impl BufferArg<'state_in, <Self::Backend as Backend>::NativeBuffer>,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _state_stride: u32,
        _suffix_len: u32,
        _in_proj_stride: u32,
        _model_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct ShortConvPrefillCpuKernel;

impl ShortConvPrefillKernel for ShortConvPrefillCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _has_bias: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'padded, 'in_proj, 'w, 'b, 'out, 'state_out, 'encoder>(
        &self,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _state_out: impl BufferArg<'state_out, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_len: u32,
        _kernel_size: u32,
        _in_proj_stride: u32,
        _state_stride: u32,
        _model_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'padded, 'in_proj, 'w, 'b, 'out, 'state_out, 'encoder, 'predicate>(
        &self,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _state_out: impl BufferArg<'state_out, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_len: u32,
        _kernel_size: u32,
        _in_proj_stride: u32,
        _state_stride: u32,
        _model_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct ShortConvDecodeCpuKernel;

impl ShortConvDecodeKernel for ShortConvDecodeCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _has_bias: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'in_proj, 'w, 'b, 'state, 'out, 'next_state, 'encoder>(
        &self,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _state: impl BufferArg<'state, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _next_state: impl BufferArg<'next_state, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_len: u32,
        _kernel_size: u32,
        _in_proj_stride: u32,
        _state_stride: u32,
        _model_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'in_proj, 'w, 'b, 'state, 'out, 'next_state, 'encoder, 'predicate>(
        &self,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _state: impl BufferArg<'state, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _next_state: impl BufferArg<'next_state, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_len: u32,
        _kernel_size: u32,
        _in_proj_stride: u32,
        _state_stride: u32,
        _model_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct ShortConvTrieCpuKernel;

impl ShortConvTrieKernel for ShortConvTrieCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _has_bias: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'in_proj, 'w, 'b, 'base_state, 'parents, 'out, 'suffix_state, 'encoder>(
        &self,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _base_state: impl BufferArg<'base_state, <Self::Backend as Backend>::NativeBuffer>,
        _parents: impl BufferArg<'parents, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_state: impl BufferArg<'suffix_state, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_len: u32,
        _kernel_size: u32,
        _in_proj_stride: u32,
        _state_stride: u32,
        _model_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<
        'in_proj,
        'w,
        'b,
        'base_state,
        'parents,
        'out,
        'suffix_state,
        'encoder,
        'predicate,
    >(
        &self,
        _in_proj: impl BufferArg<'in_proj, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _base_state: impl BufferArg<'base_state, <Self::Backend as Backend>::NativeBuffer>,
        _parents: impl BufferArg<'parents, <Self::Backend as Backend>::NativeBuffer>,
        _out: impl BufferArg<'out, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_state: impl BufferArg<'suffix_state, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_len: u32,
        _kernel_size: u32,
        _in_proj_stride: u32,
        _state_stride: u32,
        _model_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct SigmoidCpuKernel;

impl SigmoidKernel for SigmoidCpuKernel {
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
        _total_elements: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'output, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _output: impl BufferArg<'output, <Self::Backend as Backend>::NativeBuffer>,
        _total_elements: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct Conv1dPackCpuKernel;

impl Conv1dPackKernel for Conv1dPackCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'state_in, 'x, 'padded, 'encoder>(
        &self,
        _state_in: impl BufferArg<'state_in, <Self::Backend as Backend>::NativeBuffer>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _state_stride: u32,
        _row_stride: u32,
        _suffix_len: u32,
        _num_channels: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'state_in, 'x, 'padded, 'encoder, 'predicate>(
        &self,
        _state_in: impl BufferArg<'state_in, <Self::Backend as Backend>::NativeBuffer>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _state_stride: u32,
        _row_stride: u32,
        _suffix_len: u32,
        _num_channels: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct Conv1dDecodeCpuKernel;

impl Conv1dDecodeKernel for Conv1dDecodeCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _activation_type: u32,
        _has_bias: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'x, 'w, 'b, 'state, 'x_out, 'b_out, 'c_out, 'next_state, 'encoder>(
        &self,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _state: impl BufferArg<'state, <Self::Backend as Backend>::NativeBuffer>,
        _x_out: impl BufferArg<'x_out, <Self::Backend as Backend>::NativeBuffer>,
        _b_out: impl BufferArg<'b_out, <Self::Backend as Backend>::NativeBuffer>,
        _c_out: impl BufferArg<'c_out, <Self::Backend as Backend>::NativeBuffer>,
        _next_state: impl BufferArg<'next_state, <Self::Backend as Backend>::NativeBuffer>,
        _kernel_size: u32,
        _row_stride: u32,
        _state_stride: u32,
        _num_channels: u32,
        _suffix_len: u32,
        _inner_dim: u32,
        _proj_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<
        'x,
        'w,
        'b,
        'state,
        'x_out,
        'b_out,
        'c_out,
        'next_state,
        'encoder,
        'predicate,
    >(
        &self,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _state: impl BufferArg<'state, <Self::Backend as Backend>::NativeBuffer>,
        _x_out: impl BufferArg<'x_out, <Self::Backend as Backend>::NativeBuffer>,
        _b_out: impl BufferArg<'b_out, <Self::Backend as Backend>::NativeBuffer>,
        _c_out: impl BufferArg<'c_out, <Self::Backend as Backend>::NativeBuffer>,
        _next_state: impl BufferArg<'next_state, <Self::Backend as Backend>::NativeBuffer>,
        _kernel_size: u32,
        _row_stride: u32,
        _state_stride: u32,
        _num_channels: u32,
        _suffix_len: u32,
        _inner_dim: u32,
        _proj_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct Conv1dScanCpuKernel;

impl Conv1dScanKernel for Conv1dScanCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _activation_type: u32,
        _has_bias: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'padded, 'w, 'b, 'x_out, 'b_out, 'c_out, 'state_out, 'encoder>(
        &self,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _x_out: impl BufferArg<'x_out, <Self::Backend as Backend>::NativeBuffer>,
        _b_out: impl BufferArg<'b_out, <Self::Backend as Backend>::NativeBuffer>,
        _c_out: impl BufferArg<'c_out, <Self::Backend as Backend>::NativeBuffer>,
        _state_out: impl BufferArg<'state_out, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_len: u32,
        _kernel_size: u32,
        _row_stride: u32,
        _state_stride: u32,
        _num_channels: u32,
        _inner_dim: u32,
        _proj_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'padded, 'w, 'b, 'x_out, 'b_out, 'c_out, 'state_out, 'encoder, 'predicate>(
        &self,
        _padded: impl BufferArg<'padded, <Self::Backend as Backend>::NativeBuffer>,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _b: Option<impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>>,
        _x_out: impl BufferArg<'x_out, <Self::Backend as Backend>::NativeBuffer>,
        _b_out: impl BufferArg<'b_out, <Self::Backend as Backend>::NativeBuffer>,
        _c_out: impl BufferArg<'c_out, <Self::Backend as Backend>::NativeBuffer>,
        _state_out: impl BufferArg<'state_out, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_len: u32,
        _kernel_size: u32,
        _row_stride: u32,
        _state_stride: u32,
        _num_channels: u32,
        _inner_dim: u32,
        _proj_dim: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct SplitInProjCpuKernel;

impl SplitInProjKernel for SplitInProjCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'input, 'conv_out, 'z_out, 'dt_out, 'z_bias, 'encoder>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _conv_out: impl BufferArg<'conv_out, <Self::Backend as Backend>::NativeBuffer>,
        _z_out: impl BufferArg<'z_out, <Self::Backend as Backend>::NativeBuffer>,
        _dt_out: impl BufferArg<'dt_out, <Self::Backend as Backend>::NativeBuffer>,
        _z_bias: impl BufferArg<'z_bias, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_length: u32,
        _total_dim: u32,
        _conv_dim: u32,
        _inner_dim: u32,
        _num_heads: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'input, 'conv_out, 'z_out, 'dt_out, 'z_bias, 'encoder, 'predicate>(
        &self,
        _input: impl BufferArg<'input, <Self::Backend as Backend>::NativeBuffer>,
        _conv_out: impl BufferArg<'conv_out, <Self::Backend as Backend>::NativeBuffer>,
        _z_out: impl BufferArg<'z_out, <Self::Backend as Backend>::NativeBuffer>,
        _dt_out: impl BufferArg<'dt_out, <Self::Backend as Backend>::NativeBuffer>,
        _z_bias: impl BufferArg<'z_bias, <Self::Backend as Backend>::NativeBuffer>,
        _suffix_length: u32,
        _total_dim: u32,
        _conv_dim: u32,
        _inner_dim: u32,
        _num_heads: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
