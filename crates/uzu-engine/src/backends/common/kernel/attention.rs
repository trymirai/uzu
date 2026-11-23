use std::error::Error;

use crate::{
    backends::common::{Backend, BufferMut, BufferRef, CommandBuffer, Kernels, kernel::kv_layout::KVLayout},
    data_type::DataType,
    encodable_block::mixer::attention::KVCacheState,
};

pub trait AttentionKernel: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<AttentionKernel = Self>>;
    type Error: Error + Send + 'static;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        config: AttentionKernelConfig,
    ) -> Result<Self, Self::Error>;

    fn kv_layout(&self) -> KVLayout;

    fn prepare_draft_blocks(
        &self,
        _trie: Option<impl BufferRef<Backend = Self::Backend>>,
        _rows: u32,
        _prefix: u32,
        _selected: KeyBlocks<impl BufferMut<Backend = Self::Backend>>,
        _command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<<Self::Backend as Backend>::ScratchBuffer, <Self::Backend as Backend>::Error> {
        unreachable!("sparse attention is Metal only")
    }

    fn select_blocks(
        &self,
        _query_codes: impl BufferRef<Backend = Self::Backend>,
        _query_scales: impl BufferRef<Backend = Self::Backend>,
        _key_codes: impl BufferRef<Backend = Self::Backend>,
        _key_scales: impl BufferRef<Backend = Self::Backend>,
        _rows: u32,
        _prefix_len: u32,
        _cached_block_row: u32,
        _candidates: u32,
        _selected: KeyBlocks<impl BufferMut<Backend = Self::Backend>>,
        _trie: Option<impl BufferRef<Backend = Self::Backend>>,
        _command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        unreachable!("sparse attention is Metal only")
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
    pub sparse: Option<SparseAttentionKernelConfig>,
}

#[derive(Clone, Copy)]
pub struct SparseAttentionKernelConfig {
    pub num_heads: u32,
    pub head_dim: u32,
    pub compression_ratio: u32,
}

pub struct AttentionArguments<
    'a,
    B: Backend,
    TT: BufferRef<Backend = B>,
    GT: BufferRef<Backend = B>,
    ST: BufferRef<Backend = B>,
> {
    pub suffix_length: u32,
    pub trie: Option<TT>,
    pub key_blocks: Option<KeyBlocks<ST>>,
    pub sinks: Option<&'a B::GlobalGpuBuffer>,
    pub cache: KVCacheState,
    pub gate: Option<GT>,
}

pub struct KeyBlocks<T> {
    pub query_blocks: T,
    pub blocks_per_query: u32,
}

#[derive(Clone, Copy)]
pub struct AttentionInput<QT: BufferRef, KT: BufferRef, VT: BufferRef> {
    pub query_codes: QT,
    pub query_scales: QT,
    pub k_codes: KT,
    pub k_scales: KT,
    pub v_codes: VT,
    pub v_scales: VT,
}
