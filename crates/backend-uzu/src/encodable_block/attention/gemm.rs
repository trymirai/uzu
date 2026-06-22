use std::{
    cell::RefCell,
    collections::{HashMap, hash_map::Entry},
};

use crate::{
    backends::common::{
        Allocation, AsBufferRangeRef, Backend, Buffer, Context, Encoder, Kernels,
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
    pub suffix_length: usize,         // qL
    pub sequence_length: usize,       // kL (prefix + suffix)
    pub segment_prefix_length: usize, // qL_off
    pub ring_params: Option<RingParams>,
    pub head_dim: usize,
    pub sliding_window_size: Option<usize>,
    pub is_causal: bool,
    pub scale: f32,
    /// Element stride between kv-heads (groups) in the K buffer.
    pub k_head_stride: u64,
    /// Element stride between tokens in the K buffer.
    pub k_seq_stride: u64,
    /// Element stride between kv-heads (groups) in the V buffer.
    pub v_head_stride: u64,
    /// Element stride between tokens in the V buffer.
    pub v_seq_stride: u64,
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

    pub fn encode<KVBuf: AsBufferRangeRef<Buffer: Buffer<Backend = B>>>(
        &self,
        encoder: &mut Encoder<B>,
        args: AttentionGemmArguments<B, KVBuf>,
    ) -> Result<(), B::Error> {
        let head_dim = args.head_dim;
        // MXU (M5 tensor units) needs BK == 32 (KEY_GRID_COLS even) and runs a
        // 64-row Q block. Used for the f16/bf16 inference path only: MXU's f32
        // matmul is reduced-precision (fails f32's tight eps), and a 64-row f32 Q
        // block busts the 32 KB threadgroup limit at head_dim 128.
        //
        // Benchmarked on M5 (bf16): MXU is 2.2-2.7x faster than simdgroup on
        // prefill but ties/loses on decode (suffix=1) because the 64-row Q block
        // is then 1/64 utilized. So MXU only when there is at least one full Q
        // block of queries. ponytail: head_dim capped at 128 (256 busts tg-mem /
        // spills); MXU_MIN_SUFFIX is a heuristic — tune if speculative-decode
        // batch sizes land between it and the crossover.
        const MXU_MIN_SUFFIX: usize = 64;
        let use_mxu = encoder.context().supports_mxu()
            && self.data_type != DataType::F32
            && head_dim <= 128
            && args.suffix_length >= MXU_MIN_SUFFIX;
        let bk: usize = if use_mxu || args.head_dim < 128 {
            32
        } else {
            16
        };
        // The threadgroup query-block is 4 simdgroups * frag_rows rows (frag_rows =
        // 16 on MXU, 8 on simdgroup): 64 on MXU, 32 on simdgroup.
        let bq: usize = if use_mxu {
            64
        } else {
            32
        };
        let align_q = args.suffix_length.is_multiple_of(bq);
        let align_k = args.sequence_length.is_multiple_of(bk);
        let is_kv_cache_ring = args.ring_params.is_some();
        let is_causal = args.is_causal;
        let is_trie = args.trie.is_some();
        let is_sliding_window = args.sliding_window_size.is_some();
        let has_sinks = args.sinks.is_some();
        let key = KernelKey {
            bk,
            head_dim,
            use_mxu,
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
                    use_mxu,
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
        let q_head_stride = (args.suffix_length * head_dim) as u64;
        let q_seq_stride = head_dim as u64;

        let o_head_stride = head_dim as u64;
        let o_seq_stride = (args.num_heads * head_dim) as u64;

        let nk = args.sequence_length.div_ceil(bk);
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
            nk: nk as u32,
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
    use_mxu: bool,
    align_q: bool,
    align_k: bool,
    is_kv_cache_ring: bool,
    is_causal: bool,
    is_trie: bool,
    is_sliding_window: bool,
    has_sinks: bool,
}

#[cfg(test)]
#[path = "../../../unit/encodable_block/attention_gemm_test.rs"]
mod tests;
