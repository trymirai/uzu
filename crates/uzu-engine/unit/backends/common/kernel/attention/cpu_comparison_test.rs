use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding, Context, Kernels,
            gpu_types::trie::TrieNode,
            kernel::{
                AttentionArguments, AttentionInput, AttentionKernel, AttentionKernelConfig, AttentionPrepareKernel,
            },
        },
        cpu::Cpu,
    },
    data_type::{DataType, DataTypeRepr},
    encodable_block::mixer::attention::{KVCache, KVCacheView},
    tests::{
        assert::assert_eq_float,
        attention::default_attention_config,
        helpers::{
            buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend,
            submit_command_buffer,
        },
    },
};

fn run<B: Backend, T: DataTypeRepr>(
    config: AttentionKernelConfig,
    cache: KVCacheView,
    nodes: &[TrieNode],
    is_trie: bool,
) -> (Vec<T>, f32) {
    let context = create_context::<B>();
    let suffix = nodes.len() as u32;
    let prefix = cache.prefix_len();
    let dim = config.head_dim;
    let stride = (config.num_q_heads + 2 * config.num_groups) * dim;
    let values: Vec<T> =
        (0..(prefix + suffix) * stride).map(|i| T::from(((i * 17 % 251) as f32 - 125.0) / 128.0).unwrap()).collect();
    let data = create_buffer_with_data::<B, T>(&context, &values);
    let kernel = <B::Kernels as Kernels>::AttentionKernel::new(&context, config).unwrap();
    let mut command = context.create_command_buffer(None, None).unwrap();
    let mut kv = KVCache::<B>::allocate_scratch(kernel.kv_layout(), prefix + suffix, &mut command).unwrap();
    let mut q = create_buffer::<B, i8>(&context, (config.num_q_heads * suffix * dim) as usize);
    let mut qs = create_buffer::<B, bf16>(&context, (config.num_q_heads * suffix) as usize);
    let prepare = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
        &context,
        config.data_type,
        DataType::F32,
        dim,
        false,
        config.num_groups,
        kernel.kv_layout().backend_opaque_layout,
        true,
        false,
        false,
        false,
    )
    .unwrap();
    command.encode_fill(kv.k_codes.as_mut(), 0x7f);
    command.encode_fill(kv.v_codes.as_mut(), 0xff);
    command.encode_fill(kv.k_scales.as_mut(), 0xff);
    command.encode_fill(kv.v_scales.as_mut(), 0xff);
    for (start, length, heads) in [(0, prefix, 0), (prefix, suffix, config.num_q_heads)] {
        if length == 0 {
            continue;
        }
        let offset = (start * stride
            + if heads == 0 {
                config.num_q_heads * dim
            } else {
                0
            }) as usize
            * config.data_type.size_in_bytes();
        prepare.encode(
            data.subrange(offset..),
            &mut q,
            &mut qs,
            Some(kv.k_codes.as_mut()),
            Some(kv.k_scales.as_mut()),
            Some(kv.v_codes.as_mut()),
            Some(kv.v_scales.as_mut()),
            None::<&B::GlobalGpuBuffer>,
            None::<&B::GlobalGpuBuffer>,
            None::<&B::GlobalGpuBuffer>,
            None::<&B::GlobalGpuBuffer>,
            None::<&B::GlobalGpuBuffer>,
            Default::default(),
            heads,
            None,
            start,
            stride,
            length,
            heads / config.num_groups * length,
            &mut command,
        );
    }
    let trie = create_buffer_with_data::<B, u32>(&context, bytemuck::cast_slice(nodes));
    let sinks = config
        .has_sinks
        .then(|| create_buffer_with_data::<B, T>(&context, &vec![T::from(1.5).unwrap(); config.num_q_heads as usize]));
    let gate = config.gate_row_stride.map(|stride| {
        create_buffer_with_data::<B, T>(
            &context,
            &(0..suffix * stride).map(|i| T::from((i % 37) as f32 / 17.0 - 1.0).unwrap()).collect::<Vec<_>>(),
        )
    });
    let result = kernel
        .encode(
            AttentionInput {
                query_codes: &q,
                query_scales: &qs,
                k_codes: kv.k_codes.as_ref(),
                k_scales: kv.k_scales.as_ref(),
                v_codes: kv.v_codes.as_ref(),
                v_scales: kv.v_scales.as_ref(),
            },
            AttentionArguments {
                suffix_length: suffix,
                trie: is_trie.then_some(&trie),
                sinks: sinks.as_ref(),
                cache,
                gate: gate.as_ref(),
            },
            &mut command,
        )
        .unwrap();
    let mut output = create_buffer::<B, T>(&context, (suffix * config.num_q_heads * dim) as usize);
    command.encode_copy(&result, &mut output);
    submit_command_buffer(command);
    // Layouts other than the row-major one (Metal's MXU tiles) also quantize attention probabilities to u8.
    let tolerance = if kernel.kv_layout().backend_opaque_layout != 0 {
        1.5e-2
    } else {
        5e-3
    };
    (buffer_to_vec::<B, T>(&context, &output), tolerance)
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
                for_each_non_cpu_backend!(|B| {
                    let (actual, tolerance) = run::<B, bf16>(config, cache, nodes, is_trie);
                    assert_eq_float(
                        &expected,
                        &actual,
                        tolerance,
                        &format!(
                            "INT8 D{dim} causal={causal} ring={ring} window={window:?} scale={scale:?} trie={is_trie}"
                        ),
                    );
                });
            }
        }
    }
}

#[uzu_test]
fn attention_large_tree_and_f32() {
    let tree = |nodes: u32| -> Vec<[u32; 3]> {
        (0..nodes)
            .map(|i| {
                if i == 0 {
                    [0, nodes - 1, 0]
                } else {
                    [i, i, 1]
                }
            })
            .collect()
    };
    let words = tree(80);
    let nodes: &[TrieNode] = bytemuck::cast_slice(&words);
    let config = default_attention_config::<f32>(128, 8, 2, true);
    let (expected, _) = run::<Cpu, f32>(config, KVCacheView::full(37), nodes, true);
    for_each_non_cpu_backend!(|B| {
        let (actual, tolerance) = run::<B, f32>(config, KVCacheView::full(37), nodes, true);
        assert_eq_float(&expected, &actual, tolerance, "INT8 f32 branching tree with 80 nodes");
    });
    // A 1024-key cache splits the keys of 16 queries; 16 keys before 64 causal queries leave a split without keys.
    for (suffix, dim, prefix) in [(80, 64, 37), (80, 256, 37), (80, 512, 37), (16, 256, 1024), (64, 256, 16)] {
        let words = tree(suffix);
        let nodes: &[TrieNode] = bytemuck::cast_slice(&words);
        for (causal, ring, window) in
            [(true, false, None), (false, false, None), (true, true, Some(17)), (false, true, Some(17))]
        {
            let mut config = default_attention_config::<bf16>(dim, 24, 4, causal);
            config.has_sinks = true;
            config.is_kv_cache_ring = ring;
            config.sliding_window_size = window;
            let cache = if ring {
                KVCacheView::ring(prefix, 11)
            } else {
                KVCacheView::full(prefix)
            };
            for is_trie in [false, true] {
                let (expected, _) = run::<Cpu, bf16>(config, cache, nodes, is_trie);
                for_each_non_cpu_backend!(|B| {
                    let (actual, tolerance) = run::<B, bf16>(config, cache, nodes, is_trie);
                    // Metal's D512 path lands up to two BF16 steps (7.8e-3 near 0.76) from the CPU.
                    let tolerance = if dim == 512 {
                        tolerance.max(1e-2)
                    } else {
                        tolerance
                    };
                    assert_eq_float(
                        &expected,
                        &actual,
                        tolerance,
                        &format!(
                            "INT8 BF16 D{dim} {suffix} nodes prefix={prefix} causal={causal} ring={ring} \
                             window={window:?} trie={is_trie}"
                        ),
                    );
                });
            }
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
                    for_each_non_cpu_backend!(|B| {
                        let (actual, tolerance) = run::<B, bf16>(config, cache, nodes, false);
                        assert_eq_float(
                            &expected,
                            &actual,
                            tolerance,
                            &format!("single-query INT8 D{dim} Q{q_heads} prefix={prefix} ring={ring}"),
                        );
                    });
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
            for_each_non_cpu_backend!(|B| {
                let (actual, tolerance) = run::<B, f32>(config, cache, nodes, true);
                assert_eq_float(&expected, &actual, tolerance, "INT8 vector partial rows, tree, F32, sinks and gate");
            });
        }
    }
}
