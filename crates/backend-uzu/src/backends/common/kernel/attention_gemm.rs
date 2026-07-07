use crate::{
    backends::common::{
        Backend, Encoder,
        gpu_types::{attention::AttnParams, ring::RingParams},
        kernel::{BufferArg, BufferArgMut, Kernels},
    },
    data_type::DataType,
};

pub(crate) fn retile_params(
    mut params: AttnParams,
    bq: u32,
    bk: u32,
) -> AttnParams {
    params.nq_aligned = params.q_len / bq;
    params.q_rem = params.q_len % bq;
    params.nk = params.k_len.div_ceil(bk);
    params.nk_aligned = params.k_len / bk;
    params.k_rem = params.k_len % bk;
    params
}

pub struct AttentionGemmArgs<Q, K, V, O, T, S> {
    pub q: Q,
    pub k: K,
    pub v: V,
    pub o: O,
    pub params: AttnParams,
    pub ring_params: Option<RingParams>,
    pub trie: Option<T>,
    pub sliding_window_size: Option<u32>,
    pub sinks: Option<S>,
    pub num_heads: u32,
    pub suffix_length: u32,
}

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
