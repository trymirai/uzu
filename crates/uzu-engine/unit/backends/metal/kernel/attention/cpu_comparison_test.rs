use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, gpu_types::trie::TrieNode, kernel::AttentionKernelConfig},
        cpu::Cpu,
        metal::{Metal, gpu_types::kv_layout::KVLayout, kernel::attention::tests::default_attention_config},
    },
    data_type::DataTypeRepr,
    encodable_block::mixer::attention::KVCacheView,
    tests::assert::assert_eq_float,
};

fn run<B: Backend, T: DataTypeRepr>(
    config: AttentionKernelConfig,
    cache: KVCacheView,
    nodes: &[TrieNode],
    is_trie: bool,
) -> (Vec<T>, f32) {
    let (output, layout) = crate::tests::attention::run_attention::<B, T>(config, cache, nodes, is_trie);
    // MXU also quantizes attention probabilities to u8.
    let tolerance = if layout.backend_opaque_layout == KVLayout::MxuTiled as u32 {
        1.5e-2
    } else {
        5e-3
    };
    (output, tolerance)
}

#[uzu_test]
fn attention_windows_sinks_and_head_dims() {
    let nodes: &[TrieNode] = bytemuck::cast_slice::<[u32; 3], TrieNode>(&[
        [0, 7, 0],
        [1, 3, 1],
        [2, 2, 2],
        [3, 3, 2],
        [4, 7, 1],
        [5, 6, 2],
        [6, 6, 3],
        [7, 7, 2],
    ]);
    for dim in [64, 128, 256, 512] {
        for (causal, ring, window, prefix, scale) in [
            (true, false, None, 37, None),
            (true, true, None, 37, None),
            (false, true, None, 37, None),
            (false, false, None, 37, None),
            (false, false, Some(1), 37, None),
            (false, false, Some(7), 37, None),
            (false, true, Some(129), 37, None),
            (true, true, Some(0), 37, None),
            (true, false, Some(0), 37, None),
            (true, false, None, 37, Some(0.0)),
            (true, false, None, 37, Some(-0.125)),
            (true, false, Some(1), 17, None),
            (false, true, Some(17), 17, None),
            (true, true, Some(7), 17, None),
            (false, false, None, 0, None),
            (true, false, Some(1), 37, None),
            (false, true, Some(17), 37, None),
            (true, true, Some(7), 37, None),
        ] {
            let mut config = default_attention_config::<bf16>(dim, 8, 2, causal);
            config.has_sinks = window != Some(0);
            config.gate_row_stride = Some(8 * dim as u32 + 7);
            config.is_kv_cache_ring = ring;
            config.sliding_window_size = window;
            config.scale = scale;
            let cache = if ring {
                KVCacheView::ring(prefix, 11)
            } else {
                KVCacheView::full(prefix)
            };
            for is_trie in [false, true] {
                let (expected, _) = run::<Cpu, bf16>(config, cache, nodes, is_trie);
                let (actual, tolerance) = run::<Metal, bf16>(config, cache, nodes, is_trie);
                assert_eq_float(
                    &expected,
                    &actual,
                    tolerance,
                    &format!(
                        "INT8 D{dim} causal={causal} ring={ring} window={window:?} scale={scale:?} trie={is_trie}"
                    ),
                );
            }
        }
    }
}

#[uzu_test]
fn attention_large_tree_and_f32() {
    let words: Vec<[u32; 3]> = (0..80)
        .map(|i| {
            if i == 0 {
                [0, 79, 0]
            } else {
                [i, i, 1]
            }
        })
        .collect();
    let nodes: &[TrieNode] = bytemuck::cast_slice(&words);
    let config = default_attention_config::<f32>(128, 8, 2, true);
    let (expected, _) = run::<Cpu, f32>(config, KVCacheView::full(37), nodes, true);
    let (actual, tolerance) = run::<Metal, f32>(config, KVCacheView::full(37), nodes, true);
    assert_eq_float(&expected, &actual, tolerance, "INT8 f32 branching tree with 80 nodes");
    for (causal, ring, window) in
        [(true, false, None), (false, false, None), (true, true, Some(17)), (false, true, Some(17))]
    {
        let mut config = default_attention_config::<bf16>(256, 24, 4, causal);
        config.has_sinks = true;
        config.is_kv_cache_ring = ring;
        config.sliding_window_size = window;
        let cache = if ring {
            KVCacheView::ring(37, 11)
        } else {
            KVCacheView::full(37)
        };
        for is_trie in [false, true] {
            let (expected, _) = run::<Cpu, bf16>(config, cache, nodes, is_trie);
            let (actual, tolerance) = run::<Metal, bf16>(config, cache, nodes, is_trie);
            assert_eq_float(
                &expected,
                &actual,
                tolerance,
                &format!("INT8 BF16 80 nodes causal={causal} ring={ring} window={window:?} trie={is_trie}"),
            );
        }
    }
}

#[uzu_test]
fn attention_single_query() {
    let nodes: &[TrieNode] = bytemuck::cast_slice::<[u32; 3], TrieNode>(&[[0, 0, 0]]);
    for dim in [64, 128, 256, 512] {
        for q_heads in [6, 8] {
            for prefix in [0, 31, 32, 37, 1025] {
                for ring in [false, true] {
                    let mut config = default_attention_config::<bf16>(dim, q_heads, 2, true);
                    config.has_sinks = true;
                    config.gate_row_stride = Some(q_heads as u32 * dim as u32 + 7);
                    config.is_kv_cache_ring = ring;
                    config.sliding_window_size = ring.then_some(17);
                    let cache = if ring {
                        KVCacheView::ring(
                            prefix,
                            if prefix > 0 {
                                11
                            } else {
                                0
                            },
                        )
                    } else {
                        KVCacheView::full(prefix)
                    };
                    let (expected, _) = run::<Cpu, bf16>(config, cache, nodes, false);
                    let (actual, tolerance) = run::<Metal, bf16>(config, cache, nodes, false);
                    assert_eq_float(
                        &expected,
                        &actual,
                        tolerance,
                        &format!("single-query INT8 D{dim} Q{q_heads} prefix={prefix} ring={ring}"),
                    );
                }
            }
        }
    }
}

#[uzu_test]
fn attention_vector_partial_rows() {
    let nodes: &[TrieNode] = bytemuck::cast_slice::<[u32; 3], TrieNode>(&[[0, 1, 0], [1, 1, 1]]);
    for prefix in [17, 1031] {
        for ring in [false, true] {
            let mut config = default_attention_config::<f32>(128, 6, 3, true);
            config.has_sinks = true;
            config.gate_row_stride = Some(6 * 128 + 3);
            config.is_kv_cache_ring = ring;
            config.sliding_window_size = ring.then_some(19);
            let cache = if ring {
                KVCacheView::ring(prefix, 7)
            } else {
                KVCacheView::full(prefix)
            };
            let (expected, _) = run::<Cpu, f32>(config, cache, nodes, true);
            let (actual, tolerance) = run::<Metal, f32>(config, cache, nodes, true);
            assert_eq_float(&expected, &actual, tolerance, "INT8 vector partial rows, tree, F32, sinks and gate");
        }
    }
}
