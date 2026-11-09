use std::error::Error;

pub use super::attention_kernel_config::AttentionKernelConfig;
use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, Kernels, kernel::kv_layout::KVLayout},
    encodable_block::mixer::attention::KVCacheView,
};

pub trait AttentionKernel: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<AttentionKernel = Self>>;
    type Error: Error + 'static;

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
            '_,
            Self::Backend,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
        >,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<<Self::Backend as Backend>::ScratchBuffer, Self::Error>;
}

pub struct AttentionArguments<'a, B: Backend, TT: BufferRef<Backend = B>, GT: BufferRef<Backend = B>> {
    pub suffix_length: u32,
    pub trie: Option<TT>,
    pub sinks: Option<&'a B::GlobalGpuBuffer>,
    pub cache: KVCacheView,
    pub gate: Option<GT>,
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
