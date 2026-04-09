use std::{
    cell::RefCell,
    collections::{HashMap, hash_map::Entry},
};

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder, Kernels,
        gpu_types::{AttnParams, ring::RingParams},
        kernel::{AttentionGemmKernel, BufferArg, BufferArgMut},
    },
};

const BQ: usize = 32;

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

    pub fn encode<'queries, 'keys, 'values, 'output, 'trie, 'sinks, Queries, Keys, Values, Output, Trie, Sinks>(
        &self,
        context: &B::Context,
        encoder: &mut Encoder<B>,
        queries: Queries,
        keys: Keys,
        values: Values,
        output: Output,
        trie: Option<Trie>,
        sinks: Option<Sinks>,
        num_heads: usize,
        num_groups: usize,
        suffix_length: usize,
        sequence_length: usize,
        segment_prefix_length: usize,
        max_sequence_length: usize,
        ring_params: Option<RingParams>,
        head_dim: usize,
        sliding_window_size: Option<usize>,
        is_causal: bool,
        scale: f32,
    ) -> Result<(), B::Error>
    where
        Queries: BufferArg<'queries, B::Buffer>,
        Keys: BufferArg<'keys, B::Buffer>,
        Values: BufferArg<'values, B::Buffer>,
        Output: BufferArgMut<'output, B::Buffer>,
        Trie: BufferArg<'trie, B::Buffer>,
        Sinks: BufferArg<'sinks, B::Buffer>,
    {
        let bk: usize = if head_dim < 128 {
            32
        } else {
            16
        };
        let align_q = (suffix_length % BQ) == 0;
        let align_k = (sequence_length % bk) == 0;
        let is_kv_cache_ring = ring_params.is_some();
        let is_trie = trie.is_some();
        let is_sliding_window = sliding_window_size.is_some();
        let has_sinks = sinks.is_some();
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
                    context,
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
        let q_head_stride = (suffix_length * head_dim) as i64;
        let q_seq_stride = head_dim as i64;

        let kv_head_stride = (max_sequence_length * head_dim) as i64;
        let kv_seq_stride = head_dim as i64;

        let o_head_stride = head_dim as i64;
        let o_seq_stride = (num_heads * head_dim) as i64;

        let nk = (sequence_length + bk - 1) / bk;
        let nq_aligned = suffix_length / BQ;
        let nk_aligned = sequence_length / bk;

        let params = AttnParams {
            q_strides: [0, q_head_stride, q_seq_stride],
            k_strides: [0, kv_head_stride, kv_seq_stride],
            v_strides: [0, kv_head_stride, kv_seq_stride],
            o_strides: [0, o_head_stride, o_seq_stride],
            gqa_factor: (num_heads / num_groups) as i32,
            scale,
            q_len: suffix_length as i32,
            k_len: sequence_length as i32,
            q_off: segment_prefix_length as i32,
            nq_aligned: nq_aligned as i32,
            q_rem: (suffix_length - nq_aligned * BQ) as i32,
            nk: nk as i32,
            nk_aligned: nk_aligned as i32,
            k_rem: (sequence_length - nk_aligned * bk) as i32,
        };

        kernel.encode(
            queries,
            keys,
            values,
            output,
            params,
            ring_params,
            trie,
            sliding_window_size.map(|s| s as u32),
            sinks,
            num_heads as u32,
            suffix_length as u32,
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
