use std::fmt::Display;

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{gpu_types::trie::TrieNode, kernel::AttentionKernelConfig},
        cpu::Cpu,
        webgpu::WebGPU,
    },
    data_type::DataTypeRepr,
    encodable_block::mixer::attention::KVCacheView,
    tests::{assert::assert_eq_float, attention::run_attention},
};

fn compare<T: DataTypeRepr + Float + Display>(nodes: &[TrieNode]) {
    for dim in [64, 128, 256, 512] {
        for (causal, ring, window, sinks, gate, prefix, scale) in [
            (true, false, None, false, false, 0, None),
            (true, false, None, false, false, 37, None),
            (false, false, None, true, true, 37, Some(-0.125)),
            (true, true, None, true, true, 37, None),
            (true, true, Some(7), false, false, 17, None),
            (false, true, Some(17), true, true, 37, None),
            (true, false, Some(0), false, false, 37, None),
            (true, false, Some(0), true, true, 37, None),
            (true, false, Some(1), true, false, 17, Some(0.0)),
        ] {
            let config = AttentionKernelConfig {
                head_dim: dim,
                num_q_heads: 6,
                num_groups: 2,
                data_type: T::data_type(),
                is_causal: causal,
                is_kv_cache_ring: ring,
                sliding_window_size: window,
                has_sinks: sinks,
                gate_row_stride: gate.then_some(6 * dim + 7),
                scale,
            };
            let cache = if ring {
                KVCacheView::ring(prefix, 11)
            } else {
                KVCacheView::full(prefix)
            };
            for is_trie in [false, true] {
                let (expected, _) = run_attention::<Cpu, T>(config, cache, nodes, is_trie);
                let (actual, _) = run_attention::<WebGPU, T>(config, cache, nodes, is_trie);
                assert_eq_float(
                    &expected,
                    &actual,
                    5e-3,
                    &format!("WebGPU attention D{dim} causal={causal} ring={ring} window={window:?} trie={is_trie}"),
                );
            }
        }
    }
}

#[uzu_test]
fn attention_matches_cpu() {
    let nodes = bytemuck::cast_slice::<[u32; 3], TrieNode>(&[
        [0, 7, 0],
        [1, 3, 1],
        [2, 2, 2],
        [3, 3, 2],
        [4, 7, 1],
        [5, 6, 2],
        [6, 6, 3],
        [7, 7, 2],
    ]);
    compare::<bf16>(nodes);
    compare::<f32>(nodes);
}

#[uzu_test]
fn attention_single_query_matches_cpu() {
    compare::<bf16>(bytemuck::cast_slice::<[u32; 3], TrieNode>(&[[0, 0, 0]]));
}
