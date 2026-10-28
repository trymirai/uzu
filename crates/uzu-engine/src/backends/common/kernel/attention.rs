use std::error::Error;

use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, Kernels},
    data_type::DataType,
    encodable_block::mixer::attention::KVCacheView,
};

#[derive(Clone, Copy, PartialEq)]
pub enum KVFormat {
    BF16,
    INT8,
}

pub trait AttentionKernel: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<AttentionKernel = Self>>;
    type Error: Error + 'static;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        config: AttentionKernelConfig,
    ) -> Result<Self, Self::Error>;

    fn supports_int8(&self) -> bool {
        false
    }

    fn encode(
        &self,
        input: AttentionInput<
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
        >,
        arguments: AttentionArguments<
            '_,
            Self::Backend,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
        >,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<<Self::Backend as Backend>::ScratchBuffer, Self::Error>;
}

#[derive(Clone, Copy)]
pub struct AttentionKernelConfig {
    pub head_dim: u32,
    pub num_groups: u32,
    pub num_q_heads: u32,
    pub has_sinks: bool,
    pub gate_row_stride: Option<u32>,
    pub is_kv_cache_ring: bool,
    pub is_causal: bool,
    pub sliding_window_size: Option<u32>,
    pub scale: Option<f32>,
    pub data_type: DataType,
}

pub struct AttentionArguments<'a, B: Backend, TT: BufferRef<Backend = B>, GT: BufferRef<Backend = B>> {
    pub suffix_length: u32,
    pub trie: Option<TT>,
    pub sinks: Option<&'a B::GlobalGpuBuffer>,
    pub cache: KVCacheView,
    pub gate: Option<GT>,
}

#[derive(Clone, Copy)]
pub enum AttentionInput<QT: BufferRef, KT: BufferRef, VT: BufferRef> {
    FullPrecision(FullPrecisionInput<QT, KT, VT>),
    Int8(Int8Input<QT, KT, VT>),
}

#[derive(Clone, Copy)]
pub struct FullPrecisionInput<QT: BufferRef, KT: BufferRef, VT: BufferRef> {
    pub queries: QT,
    pub keys: KT,
    pub values: VT,
}

#[derive(Clone, Copy)]
pub struct Int8Input<QT: BufferRef, KT: BufferRef, VT: BufferRef> {
    pub query_codes: QT,
    pub query_scales: QT,
    pub k_codes: KT,
    pub k_scales: KT,
    pub v_codes: VT,
    pub v_scales: VT,
}
