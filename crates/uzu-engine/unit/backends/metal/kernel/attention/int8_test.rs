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
        metal::{Metal, kernel::attention::tests::default_attention_config},
    },
    data_type::{DataType, DataTypeRepr},
    encodable_block::mixer::attention::{Int8KV, KVCacheView},
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, submit_command_buffer},
    },
};

fn run<B: Backend, T: DataTypeRepr>(
    config: AttentionKernelConfig,
    cache: KVCacheView,
    nodes: &[TrieNode],
) -> Vec<T> {
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
    let mut kv = Int8KV::<B>::allocate_scratch(kernel.kv_layout(), prefix + suffix, &mut command).unwrap();
    let mut q = create_buffer::<B, i8>(&context, (config.num_q_heads * suffix * dim) as usize);
    let mut qs = create_buffer::<B, bf16>(&context, (config.num_q_heads * suffix) as usize);
    let prepare = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
        &context,
        config.data_type,
        DataType::F32,
        dim,
        false,
        config.num_groups,
        kernel.kv_layout().use_mxu,
        true,
        false,
        false,
        false,
    )
    .unwrap();
    // Poison padding to catch masked reads and partially filled packed blocks.
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
                trie: (suffix > 1).then_some(&trie),
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
    buffer_to_vec::<B, T>(&context, &output)
}

#[uzu_test]
fn int8_attention_windows_sinks_and_head_dims() {
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
        for (causal, ring, window, prefix) in [
            (true, false, None, 37),
            (true, false, Some(1), 17),
            (false, true, Some(17), 17),
            (true, true, Some(7), 17),
            (false, false, None, 0),
            (true, false, Some(1), 37),
            (false, true, Some(17), 37),
            (true, true, Some(7), 37),
        ] {
            let mut config = default_attention_config::<bf16>(dim, 8, 2, causal);
            config.has_sinks = true;
            config.gate_row_stride = Some(8 * dim as u32 + 7);
            config.is_kv_cache_ring = ring;
            config.sliding_window_size = window;
            let cache = if ring {
                KVCacheView::ring(prefix, 11)
            } else {
                KVCacheView::full(prefix)
            };
            let expected = run::<Cpu, bf16>(config, cache, nodes);
            let actual = run::<Metal, bf16>(config, cache, nodes);
            assert_eq_float(
                &expected,
                &actual,
                5e-3,
                &format!("INT8 D{dim} causal={causal} ring={ring} window={window:?}"),
            );
        }
    }
}

#[uzu_test]
fn int8_attention_large_tree_and_f32() {
    // Root plus sibling leaves: exercises the generic mask beyond the 64-bit fast path.
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
    let expected = run::<Cpu, f32>(config, KVCacheView::full(37), nodes);
    let actual = run::<Metal, f32>(config, KVCacheView::full(37), nodes);
    assert_eq_float(&expected, &actual, 5e-3, "INT8 f32 branching tree with 80 nodes");
    let mut config = default_attention_config::<bf16>(256, 24, 4, true);
    config.has_sinks = true;
    let expected = run::<Cpu, bf16>(config, KVCacheView::full(37), nodes);
    let actual = run::<Metal, bf16>(config, KVCacheView::full(37), nodes);
    assert_eq_float(&expected, &actual, 5e-3, "INT8 BF16 branching tree with 80 nodes");
}

#[uzu_test]
fn int8_attention_single_query() {
    let nodes: &[TrieNode] = bytemuck::cast_slice::<[u32; 3], TrieNode>(&[[0, 0, 0]]);
    for dim in [64, 128, 256, 512] {
        for q_heads in [6, 8] {
            for prefix in [0, 37, 1025] {
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
                    let expected = run::<Cpu, bf16>(config, cache, nodes);
                    let actual = run::<Metal, bf16>(config, cache, nodes);
                    assert_eq_float(
                        &expected,
                        &actual,
                        5e-3,
                        &format!("single-query INT8 D{dim} Q{q_heads} prefix={prefix} ring={ring}"),
                    );
                }
            }
        }
    }
}

#[uzu_test]
fn int8_attention_vector_partial_rows() {
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
            let expected = run::<Cpu, f32>(config, cache, nodes);
            let actual = run::<Metal, f32>(config, cache, nodes);
            assert_eq_float(&expected, &actual, 5e-3, "INT8 vector partial rows, tree, F32, sinks and gate");
        }
    }
}
