use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            BufferArg, BufferArgMut, Encoder,
            gpu_types::trie::TrieNode,
            kernel::{
                AttentionGemmKernel,
                attention_gemm::{AttentionGemmArgs, AttentionGemmDispatch, retile_params},
            },
        },
        cpu::{Cpu, context::CpuContext, error::CpuError, kernel::attention::mask::should_use_key},
    },
    data_type::DataType,
};

#[kernel(AttentionGemm)]
#[variants(T, f32, f16, bf16)]
#[variants(BK, 16, 32)]
#[variants(BD, 64, 128, 256)]
#[variants(USE_MXU, false)]
pub fn attention_gemm<T: ArrayElement + Float, const BK: u32, const BD: u32, const USE_MXU: bool>(
    q: *const T,
    k: *const T,
    v: *const T,
    o: *mut T,
    params: crate::backends::common::gpu_types::attention::AttnParams,
    #[optional(is_kv_cache_ring)] ring_params: Option<crate::backends::common::gpu_types::ring::RingParams>,
    #[optional(is_trie)] trie: Option<*const TrieNode>,
    #[optional(is_sliding_window)] sliding_window_size: Option<u32>,
    #[optional(has_sinks)] sinks: Option<*const T>,
    num_heads: u32,
    suffix_length: u32,
    #[specialize] align_q: bool,
    #[specialize] align_k: bool,
    #[specialize] is_kv_cache_ring: bool,
    #[specialize] is_causal: bool,
    #[specialize] is_trie: bool,
    #[specialize] is_sliding_window: bool,
    #[specialize] has_sinks: bool,
) {
    assert_eq!(suffix_length, params.q_len);
    assert_eq!(align_q, params.q_rem == 0);
    assert_eq!(align_k, params.k_rem == 0);
    assert_eq!(sliding_window_size.is_some(), is_sliding_window);

    let q_len = params.q_len as usize;
    let k_len = params.k_len as usize;
    let head_dim = BD as usize;
    let prefix_length = params.q_off as usize;
    let suffix_position = if is_kv_cache_ring {
        ring_params.unwrap().ring_length as usize
    } else {
        prefix_length
    };

    for head_idx in 0..num_heads as usize {
        let kv_head_idx = head_idx / params.gqa_factor as usize;

        let q_head = unsafe { q.add(head_idx * params.q_strides[1] as usize) };
        let k_head = unsafe { k.add(kv_head_idx * params.k_strides[1] as usize) };
        let v_head = unsafe { v.add(kv_head_idx * params.v_strides[1] as usize) };
        let o_head = unsafe { o.add(head_idx * params.o_strides[1] as usize) };

        for qi in 0..q_len {
            let q_row = unsafe { q_head.add(qi * params.q_strides[2] as usize) };
            let o_row = unsafe { o_head.add(qi * params.o_strides[2] as usize) };

            let query_position = if is_trie {
                let trie_node = unsafe { &*trie.unwrap().add(qi) };
                suffix_position + trie_node.height as usize
            } else {
                suffix_position + qi
            };

            // Read query row and pre-scale
            let mut q_vec = vec![0.0f32; head_dim];
            for j in 0..head_dim {
                q_vec[j] = params.scale * unsafe { *q_row.add(j) }.to_f32().unwrap();
            }

            let mut max_score = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;
            let mut o_acc = vec![0.0f32; head_dim];

            // Initialize with attention sinks if present
            if has_sinks {
                max_score = unsafe { *sinks.unwrap().add(head_idx) }.to_f32().unwrap();
                sum_exp = 1.0;
            }

            // Loop over all key positions
            for ki in 0..k_len {
                if !should_use_key(
                    ring_params,
                    trie,
                    sliding_window_size,
                    qi as u32,
                    prefix_length as u32,
                    suffix_position as u32,
                    query_position as u32,
                    ki as u32,
                    is_causal,
                ) {
                    continue;
                }

                // Compute dot product: score = q . k[ki]
                let k_row = unsafe { k_head.add(ki * params.k_strides[2] as usize) };
                let mut score = 0.0f32;
                for j in 0..head_dim {
                    score += q_vec[j] * unsafe { *k_row.add(j) }.to_f32().unwrap();
                }

                // Online softmax update
                let new_max = f32::max(max_score, score);
                let factor = (max_score - new_max).exp();
                let exp_score = (score - new_max).exp();

                max_score = new_max;
                sum_exp = sum_exp * factor + exp_score;

                // Update output accumulator
                let v_row = unsafe { v_head.add(ki * params.v_strides[2] as usize) };
                for j in 0..head_dim {
                    o_acc[j] = o_acc[j] * factor + exp_score * unsafe { *v_row.add(j) }.to_f32().unwrap();
                }
            }

            // Normalize and write output
            let inv_sum = 1.0 / sum_exp;
            for j in 0..head_dim {
                unsafe {
                    *o_row.add(j) = T::from(o_acc[j] * inv_sum).unwrap();
                }
            }
        }
    }
}

pub struct AttentionGemmCpuDispatch {
    tiles: [Option<AttentionGemmCpuKernel>; 4],
    data_type: DataType,
    bk: u32,
    bd: u32,
    is_kv_cache_ring: bool,
    is_causal: bool,
    is_trie: bool,
    is_sliding_window: bool,
    has_sinks: bool,
}

impl AttentionGemmCpuDispatch {
    fn get_or_create(
        &mut self,
        context: &CpuContext,
        align_q: bool,
        align_k: bool,
    ) -> Result<&AttentionGemmCpuKernel, CpuError> {
        let index = (usize::from(align_q) << 1) | usize::from(align_k);
        if self.tiles[index].is_none() {
            self.tiles[index] = Some(AttentionGemmCpuKernel::new(
                context,
                self.data_type,
                self.bk,
                self.bd,
                false,
                align_q,
                align_k,
                self.is_kv_cache_ring,
                self.is_causal,
                self.is_trie,
                self.is_sliding_window,
                self.has_sinks,
            )?);
        }
        Ok(self.tiles[index].as_ref().expect("tile was just initialized"))
    }
}

impl AttentionGemmDispatch for AttentionGemmCpuDispatch {
    type Backend = Cpu;

    fn new(
        _context: &CpuContext,
        data_type: DataType,
        bk: u32,
        bd: u32,
        is_kv_cache_ring: bool,
        is_causal: bool,
        is_trie: bool,
        is_sliding_window: bool,
        has_sinks: bool,
    ) -> Result<Self, CpuError> {
        Ok(Self {
            tiles: std::array::from_fn(|_| None),
            data_type,
            bk,
            bd,
            is_kv_cache_ring,
            is_causal,
            is_trie,
            is_sliding_window,
            has_sinks,
        })
    }

    fn encode<'q, 'k, 'v, 'o, 'trie, 'sinks>(
        &mut self,
        args: AttentionGemmArgs<
            impl BufferArg<'q, Cpu>,
            impl BufferArg<'k, Cpu>,
            impl BufferArg<'v, Cpu>,
            impl BufferArgMut<'o, Cpu>,
            impl BufferArg<'trie, Cpu>,
            impl BufferArg<'sinks, Cpu>,
        >,
        encoder: &mut Encoder<Cpu>,
    ) -> Result<(), CpuError> {
        let params = retile_params(args.params, 32, self.bk);
        let kernel = self.get_or_create(encoder.context(), params.q_rem == 0, params.k_rem == 0)?;
        kernel.encode(
            args.q,
            args.k,
            args.v,
            args.o,
            params,
            args.ring_params,
            args.trie,
            args.sliding_window_size,
            args.sinks,
            args.num_heads,
            args.suffix_length,
            encoder,
        );
        Ok(())
    }
}
