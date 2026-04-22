use std::{
    cell::RefCell,
    collections::{HashMap, hash_map::Entry},
};

use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::{AttnParams, ring::RingParams},
        kernel::AttentionGemmKernel,
    },
};

const BQ: usize = 32;

pub struct AttentionGemmArguments<'a, B: Backend> {
    pub queries_buffer: &'a Allocation<B>,
    pub keys_buffer: &'a Allocation<B>,
    pub values_buffer: &'a Allocation<B>,
    pub output_buffer: &'a mut Allocation<B>,
    pub trie_buffer: Option<&'a Allocation<B>>,
    pub sinks_buffer: Option<&'a Allocation<B>>,
    pub num_heads: usize,
    pub num_groups: usize,
    pub suffix_length: usize,         // qL
    pub sequence_length: usize,       // kL (prefix + suffix)
    pub segment_prefix_length: usize, // qL_off
    pub max_sequence_length: usize,   // stride for K/V cache
    pub ring_params: Option<RingParams>,
    pub head_dim: usize,
    pub sliding_window_size: Option<usize>,
    pub is_causal: bool,
    pub scale: f32,
}

pub struct AttentionGemmBlock<B: Backend> {
    data_type: DataType,
    cache: RefCell<HashMap<KernelKey, <B::Kernels as Kernels>::AttentionGemmKernel>>,
}

impl<B: Backend> AttentionGemmBlock<B> {
    pub fn new(data_type: DataType) -> Self {
        let cache = RefCell::new(HashMap::new());
        Self {
            data_type,
            cache,
        }
    }

    pub fn encode(
        &self,
        encoder: &mut Encoder<B>,
        args: AttentionGemmArguments<B>,
    ) -> Result<(), B::Error> {
        let bk: usize = if args.head_dim < 128 {
            32
        } else {
            16
        };
        let head_dim = args.head_dim;
        let align_q = (args.suffix_length % BQ) == 0;
        let align_k = (args.sequence_length % bk) == 0;
        let is_kv_cache_ring = args.ring_params.is_some();
        let is_causal = args.is_causal;
        let is_trie = args.trie_buffer.is_some();
        let is_sliding_window = args.sliding_window_size.is_some();
        let has_sinks = args.sinks_buffer.is_some();
        let key = KernelKey {
            bk,
            head_dim,
            align_q,
            align_k,
            is_kv_cache_ring,
            is_causal,
            is_trie,
            is_sliding_window,
            has_sinks,
        };

        let mut map = self.cache.borrow_mut();
        let kernel = match map.entry(key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let kernel = <B::Kernels as Kernels>::AttentionGemmKernel::new(
                    encoder.context(),
                    self.data_type,
                    bk as u32,
                    head_dim as u32,
                    align_q,
                    align_k,
                    is_kv_cache_ring,
                    is_causal,
                    is_trie,
                    is_sliding_window,
                    has_sinks,
                )?;
                entry.insert(kernel)
            },
        };

        // Params (all strides in elements)
        let q_head_stride = (args.suffix_length * head_dim) as i64;
        let q_seq_stride = head_dim as i64;

        let kv_head_stride = (args.max_sequence_length * head_dim) as i64;
        let kv_seq_stride = head_dim as i64;

        let o_head_stride = head_dim as i64;
        let o_seq_stride = (args.num_heads * head_dim) as i64;

        let nk = (args.sequence_length + bk - 1) / bk;
        let nq_aligned = args.suffix_length / BQ;
        let nk_aligned = args.sequence_length / bk;

        let params = AttnParams {
            q_strides: [0, q_head_stride, q_seq_stride],
            k_strides: [0, kv_head_stride, kv_seq_stride],
            v_strides: [0, kv_head_stride, kv_seq_stride],
            o_strides: [0, o_head_stride, o_seq_stride],
            gqa_factor: (args.num_heads / args.num_groups) as i32,
            scale: args.scale,
            q_len: args.suffix_length as i32,
            k_len: args.sequence_length as i32,
            q_off: args.segment_prefix_length as i32,
            nq_aligned: nq_aligned as i32,
            q_rem: (args.suffix_length - nq_aligned * BQ) as i32,
            nk: nk as i32,
            nk_aligned: nk_aligned as i32,
            k_rem: (args.sequence_length - nk_aligned * bk) as i32,
        };

        kernel.encode(
            args.queries_buffer,
            args.keys_buffer,
            args.values_buffer,
            args.output_buffer,
            params,
            args.ring_params,
            args.trie_buffer,
            args.sliding_window_size.map(|s| s as u32),
            args.sinks_buffer,
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
    align_q: bool,
    align_k: bool,
    is_kv_cache_ring: bool,
    is_causal: bool,
    is_trie: bool,
    is_sliding_window: bool,
    has_sinks: bool,
}
