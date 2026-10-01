use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, Kernels, gpu_types::trie::TrieNode},
    data_type::DataType,
    encodable_block::mixer::attention::KVCacheView,
};

pub trait AttentionKernel: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<AttentionKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        config: AttentionKernelConfig,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            Self::Backend,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
        >,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<<Self::Backend as Backend>::ScratchBuffer, <Self::Backend as Backend>::Error>;
}

#[derive(Clone, Copy)]
pub struct AttentionKernelConfig {
    pub head_dim: u32,
    pub num_groups: u32,
    pub num_q_heads: u32,
    pub has_sinks: bool,
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
> {
    pub queries: QT,
    pub keys: KT,
    pub values: VT,
    pub suffix_length: u32,
    pub trie: Option<TT>,
    pub sinks: Option<&'a B::GlobalBuffer>,
    pub cache: KVCacheView,
}

impl<
    B: Backend,
    QT: BufferRef<Backend = B>,
    TT: BufferRef<Backend = B>,
    KT: BufferRef<Backend = B>,
    VT: BufferRef<Backend = B>,
> AttentionArguments<'_, B, QT, TT, KT, VT>
{
    pub fn validate_trie_size(&self) {
        assert!(
            self.trie.is_none_or(|trie| trie.size() >= self.suffix_length as usize * size_of::<TrieNode>()),
            "trie buffer must cover every suffix token"
        );
    }
}
