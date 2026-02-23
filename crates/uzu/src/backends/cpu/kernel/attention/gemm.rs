use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{AttentionGemmKernel, BufferArg},
        },
        cpu::Cpu,
    },
};

pub struct AttentionGemmCpuKernel;

impl AttentionGemmKernel for AttentionGemmCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _BK: u32,
        _BD: u32,
        _align_q: bool,
        _align_k: bool,
        _do_causal: bool,
        _has_mask: bool,
        _has_sinks: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'q, 'k, 'v, 'o, 'mask, 'sinks, 'encoder>(
        &self,
        _q: impl BufferArg<'q, <Self::Backend as Backend>::NativeBuffer>,
        _k: impl BufferArg<'k, <Self::Backend as Backend>::NativeBuffer>,
        _v: impl BufferArg<'v, <Self::Backend as Backend>::NativeBuffer>,
        _o: impl BufferArg<'o, <Self::Backend as Backend>::NativeBuffer>,
        _params: crate::backends::common::gpu_types::attention::AttnParams,
        _mask_params: Option<crate::backends::common::gpu_types::attention::AttnMaskParams>,
        _mask: Option<impl BufferArg<'mask, <Self::Backend as Backend>::NativeBuffer>>,
        _sinks: Option<impl BufferArg<'sinks, <Self::Backend as Backend>::NativeBuffer>>,
        _num_heads: u32,
        _suffix_length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'q, 'k, 'v, 'o, 'mask, 'sinks, 'encoder, 'predicate>(
        &self,
        _q: impl BufferArg<'q, <Self::Backend as Backend>::NativeBuffer>,
        _k: impl BufferArg<'k, <Self::Backend as Backend>::NativeBuffer>,
        _v: impl BufferArg<'v, <Self::Backend as Backend>::NativeBuffer>,
        _o: impl BufferArg<'o, <Self::Backend as Backend>::NativeBuffer>,
        _params: crate::backends::common::gpu_types::attention::AttnParams,
        _mask_params: Option<crate::backends::common::gpu_types::attention::AttnMaskParams>,
        _mask: Option<impl BufferArg<'mask, <Self::Backend as Backend>::NativeBuffer>>,
        _sinks: Option<impl BufferArg<'sinks, <Self::Backend as Backend>::NativeBuffer>>,
        _num_heads: u32,
        _suffix_length: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
