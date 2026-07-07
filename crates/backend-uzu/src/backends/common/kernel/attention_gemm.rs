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

pub(crate) fn tile_variant_index(
    align_q: bool,
    align_k: bool,
) -> usize {
    (usize::from(align_q) << 1) | usize::from(align_k)
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

    #[allow(clippy::too_many_arguments)]
    fn encode<'q, 'k, 'v, 'o, 'trie, 'sinks, 'encoder>(
        &mut self,
        q: impl BufferArg<'q, Self::Backend>,
        k: impl BufferArg<'k, Self::Backend>,
        v: impl BufferArg<'v, Self::Backend>,
        o: impl BufferArgMut<'o, Self::Backend>,
        params: AttnParams,
        ring_params: Option<RingParams>,
        trie: Option<impl BufferArg<'trie, Self::Backend>>,
        sliding_window_size: Option<u32>,
        sinks: Option<impl BufferArg<'sinks, Self::Backend>>,
        num_heads: u32,
        suffix_length: u32,
        encoder: &'encoder mut Encoder<Self::Backend>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}
