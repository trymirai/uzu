use std::error::Error;

use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, Kernels, kernel::kv_layout::KVLayout},
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

    fn encode(
        &self,
        input: AttentionInput<
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
        >,
        arguments: AttentionArguments<
            Self::Backend,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
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
    /// Elements between the sink logits of consecutive queries; zero shares one learned sink per head.
    pub sink_row_stride: u32,
    pub gate_row_stride: Option<u32>,
    pub is_kv_cache_ring: bool,
    pub is_causal: bool,
    pub sliding_window_size: Option<u32>,
    pub scale: Option<f32>,
    pub data_type: DataType,
    pub sparse_compression_ratio: Option<u32>,
    /// Logits get a bias by query-to-key distance below this extent, see `AttentionArguments::relative_logits`.
    pub relative_extent: Option<u32>,
}

pub struct AttentionArguments<
    B: Backend,
    TT: BufferRef<Backend = B>,
    GT: BufferRef<Backend = B>,
    ST: BufferRef<Backend = B>,
    SK: BufferRef<Backend = B>,
    RT: BufferRef<Backend = B>,
> {
    pub suffix_length: u32,
    pub trie: Option<TT>,
    pub key_blocks: Option<KeyBlocks<ST>>,
    /// Sink logits with a zero value: one per head, or one per query and head `sink_row_stride` elements apart.
    pub sinks: Option<SK>,
    pub cache: KVCacheState,
    pub gate: Option<GT>,
    /// `[q head][query][relative_extent]` bf16 biases of the logits of keys `distance` positions before the query.
    pub relative_logits: Option<RT>,
}

pub struct KeyBlocks<ST: BufferRef> {
    pub query_blocks: ST,
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
