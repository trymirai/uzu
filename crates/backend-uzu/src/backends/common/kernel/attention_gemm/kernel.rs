use crate::{
    backends::common::{
        Backend, BufferArg, BufferArgMut, Encoder,
        kernel::{Kernels, attention_gemm::AttentionGemmArgs},
    },
    data_type::DataType,
};

pub trait AttentionGemmDispatch: Sized {
    type Backend: Backend<Kernels: Kernels<AttentionGemmDispatch = Self>>;

    #[allow(clippy::too_many_arguments)]
    fn new(
        context: &<Self::Backend as Backend>::Context,
        data_type: DataType,
        bk: u32,
        bd: u32,
        is_kv_cache_ring: bool,
        is_causal: bool,
        is_trie: bool,
        is_sliding_window: bool,
        has_sinks: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn encode<'q, 'k, 'v, 'o, 'trie, 'sinks>(
        &mut self,
        args: AttentionGemmArgs<
            impl BufferArg<'q, Self::Backend>,
            impl BufferArg<'k, Self::Backend>,
            impl BufferArg<'v, Self::Backend>,
            impl BufferArgMut<'o, Self::Backend>,
            impl BufferArg<'trie, Self::Backend>,
            impl BufferArg<'sinks, Self::Backend>,
        >,
        encoder: &mut Encoder<Self::Backend>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}
