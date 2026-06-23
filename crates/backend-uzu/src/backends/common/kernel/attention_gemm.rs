use std::{
    cell::RefCell,
    collections::{HashMap, hash_map::Entry},
};

use crate::{
    backends::common::{
        Allocation, AsBufferRangeRef, Backend, Buffer, Encoder, Kernels,
        gpu_types::{AttnParams, ring::RingParams},
        kernel::AttentionGemmKernel,
    },
    data_type::DataType,
};

pub struct AttentionGemmArguments<'a, B: Backend, KVBuf: AsBufferRangeRef> {
    pub queries: &'a Allocation<B>,
    pub keys: &'a KVBuf,
    pub values: &'a KVBuf,
    pub output: &'a mut Allocation<B>,
    pub trie: Option<&'a Allocation<B>>,
    pub sinks: Option<&'a Allocation<B>>,
    pub num_heads: usize,
    pub num_groups: usize,
    pub suffix_length: usize,
    pub sequence_length: usize,
    pub segment_prefix_length: usize,
    pub ring_params: Option<RingParams>,
    pub head_dim: usize,
    pub sliding_window_size: Option<usize>,
    pub is_causal: bool,
    pub scale: f32,
    pub k_head_stride: u64,
    pub k_seq_stride: u64,
    pub v_head_stride: u64,
    pub v_seq_stride: u64,
}

pub trait AttentionGemmBackendBlock: Sized {
    type Backend: Backend;

    fn new(data_type: DataType) -> Self;

    fn encode<KVBuf: AsBufferRangeRef<Buffer: Buffer<Backend = Self::Backend>>>(
        &self,
        encoder: &mut Encoder<Self::Backend>,
        args: AttentionGemmArguments<Self::Backend, KVBuf>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}

pub struct GeneratedAttentionGemmBlock<K: Kernels> {
    data_type: DataType,
    cache: RefCell<HashMap<KernelKey, K::AttentionGemmKernel>>,
}

impl<K: Kernels> GeneratedAttentionGemmBlock<K> {
    pub fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            cache: RefCell::new(HashMap::new()),
        }
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn encode_with_accelerator<KVBuf: AsBufferRangeRef<Buffer: Buffer<Backend = K::Backend>>>(
        &self,
        use_accelerator: bool,
        encoder: &mut Encoder<K::Backend>,
        args: AttentionGemmArguments<K::Backend, KVBuf>,
    ) -> Result<(), <K::Backend as Backend>::Error> {
        let head_dim = args.head_dim;
        let bk = if use_accelerator || head_dim < 128 {
            32
        } else {
            16
        };
        let bq = if use_accelerator {
            64
        } else {
            32
        };
        let align_q = args.suffix_length.is_multiple_of(bq);
        let align_k = args.sequence_length.is_multiple_of(bk);
        let key = KernelKey {
            bk,
            head_dim,
            use_accelerator,
            align_q,
            align_k,
            is_kv_cache_ring: args.ring_params.is_some(),
            is_causal: args.is_causal,
            is_trie: args.trie.is_some(),
            is_sliding_window: args.sliding_window_size.is_some(),
            has_sinks: args.sinks.is_some(),
        };

        let mut map = self.cache.borrow_mut();
        let kernel = match map.entry(key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(K::AttentionGemmKernel::new(
                encoder.context(),
                self.data_type,
                bk as u32,
                head_dim as u32,
                use_accelerator,
                align_q,
                align_k,
                key.is_kv_cache_ring,
                key.is_causal,
                key.is_trie,
                key.is_sliding_window,
                key.has_sinks,
            )?),
        };

        let q_head_stride = (args.suffix_length * head_dim) as u64;
        let q_seq_stride = head_dim as u64;
        let o_head_stride = head_dim as u64;
        let o_seq_stride = (args.num_heads * head_dim) as u64;
        let nq_aligned = args.suffix_length / bq;
        let nk_aligned = args.sequence_length / bk;
        let params = AttnParams {
            q_strides: [0, q_head_stride, q_seq_stride],
            k_strides: [0, args.k_head_stride, args.k_seq_stride],
            v_strides: [0, args.v_head_stride, args.v_seq_stride],
            o_strides: [0, o_head_stride, o_seq_stride],
            gqa_factor: (args.num_heads / args.num_groups) as u32,
            scale: args.scale,
            q_len: args.suffix_length as u32,
            k_len: args.sequence_length as u32,
            q_off: args.segment_prefix_length as u32,
            nq_aligned: nq_aligned as u32,
            q_rem: (args.suffix_length - nq_aligned * bq) as u32,
            nk: args.sequence_length.div_ceil(bk) as u32,
            nk_aligned: nk_aligned as u32,
            k_rem: (args.sequence_length - nk_aligned * bk) as u32,
        };

        kernel.encode(
            args.queries,
            args.keys,
            args.values,
            args.output,
            params,
            args.ring_params,
            args.trie,
            args.sliding_window_size.map(|s| s as u32),
            args.sinks,
            args.num_heads as u32,
            args.suffix_length as u32,
            encoder,
        );

        Ok(())
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct KernelKey {
    bk: usize,
    head_dim: usize,
    use_accelerator: bool,
    align_q: bool,
    align_k: bool,
    is_kv_cache_ring: bool,
    is_causal: bool,
    is_trie: bool,
    is_sliding_window: bool,
    has_sinks: bool,
}
