use std::error::Error;

use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, Kernels},
    data_type::DataType,
    encodable_block::mixer::attention::{KVCacheView, kv_storage::Int8KV},
};

#[derive(Clone, Copy, PartialEq)]
pub enum KVFormat {
    BF16,
    Int8,
}

pub trait AttentionKernel: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<AttentionKernel = Self>>;
    type Error: Error + 'static;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        config: AttentionKernelConfig,
    ) -> Result<Self, Self::Error>;

    fn supports_int8_kv(&self) -> bool {
        false
    }

    fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            Self::Backend,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
        >,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<<Self::Backend as Backend>::ScratchBuffer, Self::Error>;

    fn encode_int8(
        &self,
        _arguments: Int8AttentionArguments<
            '_,
            Self::Backend,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
        >,
        _command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<<Self::Backend as Backend>::ScratchBuffer, Self::Error> {
        unreachable!("no INT8 KV path")
    }
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

pub struct AttentionArguments<
    'a,
    B: Backend,
    QT: BufferRef<Backend = B>,
    TT: BufferRef<Backend = B>,
    KT: BufferRef<Backend = B>,
    VT: BufferRef<Backend = B>,
    GT: BufferRef<Backend = B>,
> {
    pub queries: QT,
    pub keys: KT,
    pub values: VT,
    pub suffix_length: u32,
    pub trie: Option<TT>,
    pub sinks: Option<&'a B::GlobalBuffer>,
    pub cache: KVCacheView,
    pub gate: Option<GT>,
}

pub struct Int8AttentionArguments<
    'a,
    B: Backend,
    QT: BufferRef<Backend = B>,
    ST: BufferRef<Backend = B>,
    TT: BufferRef<Backend = B>,
    GT: BufferRef<Backend = B>,
> {
    pub query_codes: QT,
    pub query_scales: ST,
    pub kv: &'a Int8KV<B>,
    pub suffix_length: u32,
    pub trie: Option<TT>,
    pub cache: KVCacheView,
    pub gate: Option<GT>,
}
