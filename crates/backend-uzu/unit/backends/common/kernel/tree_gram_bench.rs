#![cfg(metal_backend)]

use std::{mem::size_of, time::Duration};

use criterion::{BenchmarkId, Criterion};
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    array::ArrayContextExt,
    backends::{
        common::{Allocation, Backend, Context, Kernels, kernel::BuildTreeGramKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::{cold_pool::ColdPool, matmul::iter_encode_loop_named},
};

const K_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const HEAD_K_DIM: usize = 128;
const HEAD_V_DIM: usize = 128;
const TREE_SIZES: &[usize] = &[33, 49, 64, 128, 256, 512];
const BATCH_SIZES: &[usize] = &[1, 2, 4, 8];

struct TreeGramBuffers {
    q: Allocation<Metal>,
    k: Allocation<Metal>,
    trie: Allocation<Metal>,
    prefix: Allocation<Metal>,
    beta: Allocation<Metal>,
    h0: Allocation<Metal>,
    h0_idx: Allocation<Metal>,
    a_packed: Allocation<Metal>,
    qkd: Allocation<Metal>,
    a_inv: Allocation<Metal>,
    kh0: Allocation<Metal>,
}

fn reranker_like_trie(
    batch_size: usize,
    tree_size: usize,
) -> Vec<u32> {
    let mut trie = Vec::with_capacity(batch_size * tree_size * 3);
    for b in 0..batch_size {
        let mut children = vec![Vec::<usize>::new(); tree_size];
        let mut child_count = vec![0usize; tree_size];
        for node in 1..tree_size {
            let mut parent = (node - 1) / 2;
            if child_count[parent] >= 2 {
                parent = node.saturating_sub(2 + (b + node) % node);
            }
            children[parent].push(node);
            child_count[parent] += 1;
        }

        fn dfs(
            node: usize,
            children: &[Vec<usize>],
            trie: &mut [[u32; 3]],
            next: &mut u32,
            depth: u32,
        ) {
            let start = *next;
            *next += 1;
            let mut end = start;
            for &child in &children[node] {
                dfs(child, children, trie, next, depth + 1);
                end = end.max(trie[child][1]);
            }
            trie[node] = [start, end, depth];
        }

        let mut batch_trie = vec![[0u32; 3]; tree_size];
        let mut next = 0;
        dfs(0, &children, &mut batch_trie, &mut next, 0);
        for node in batch_trie {
            trie.extend_from_slice(&node);
        }
    }
    trie
}

fn make_buffers(
    context: &<Metal as Backend>::Context,
    batch_size: usize,
    tree_size: usize,
) -> (TreeGramBuffers, f32) {
    let qk_len = batch_size * tree_size * K_HEADS * HEAD_K_DIM;
    let scalar_len = batch_size * tree_size * VALUE_HEADS;
    let out_len = batch_size * VALUE_HEADS * tree_size * tree_size;
    let num_blocks = tree_size.div_ceil(16);
    let a_len = batch_size * VALUE_HEADS * num_blocks * num_blocks.div_ceil(2) * 16 * 32;
    let a_inv_len = batch_size * VALUE_HEADS * num_blocks * 16 * 16;
    let h0_len = batch_size * VALUE_HEADS * HEAD_V_DIM * HEAD_K_DIM;
    let kh0_len = batch_size * tree_size * VALUE_HEADS * HEAD_V_DIM;
    let trie = reranker_like_trie(batch_size, tree_size);
    let q = (0..qk_len).map(|i| bf16::from_f32(((i as f32 * 0.017).sin() * 0.2) + 0.01)).collect::<Vec<_>>();
    let k = (0..qk_len).map(|i| bf16::from_f32(((i as f32 * 0.019).cos() * 0.18) - 0.02)).collect::<Vec<_>>();
    let prefix = (0..scalar_len)
        .map(|i| -((i % tree_size) as f32) * 0.01 - ((i % VALUE_HEADS) as f32) * 0.003)
        .collect::<Vec<_>>();
    let beta = (0..scalar_len).map(|i| 0.25 + ((i as f32 * 0.013).sin() + 1.0) * 0.2).collect::<Vec<_>>();
    let h0 = (0..h0_len).map(|i| bf16::from_f32(((i as f32 * 0.007).sin() * 0.05) - 0.01)).collect::<Vec<_>>();
    let h0_idx = (0..batch_size).map(|i| i as i32).collect::<Vec<_>>();
    let scale = (HEAD_K_DIM as f32).sqrt().recip();
    (
        TreeGramBuffers {
            q: context.create_array_from(&[q.len()], &q).into_allocation(),
            k: context.create_array_from(&[k.len()], &k).into_allocation(),
            trie: context.create_array_from(&[trie.len()], &trie).into_allocation(),
            prefix: context.create_array_from(&[prefix.len()], &prefix).into_allocation(),
            beta: context.create_array_from(&[beta.len()], &beta).into_allocation(),
            h0: context.create_array_from(&[h0.len()], &h0).into_allocation(),
            h0_idx: context.create_array_from(&[h0_idx.len()], &h0_idx).into_allocation(),
            a_packed: context.create_array_uninitialized(&[a_len], DataType::F32).into_allocation(),
            qkd: context.create_array_uninitialized(&[out_len], DataType::F32).into_allocation(),
            a_inv: context.create_array_uninitialized(&[a_inv_len], DataType::F32).into_allocation(),
            kh0: context.create_array_uninitialized(&[kh0_len], DataType::F32).into_allocation(),
        },
        scale,
    )
}

fn buffers_bytes(
    batch_size: usize,
    tree_size: usize,
) -> usize {
    let qk_len = batch_size * tree_size * K_HEADS * HEAD_K_DIM;
    let scalar_len = batch_size * tree_size * VALUE_HEADS;
    let out_len = batch_size * VALUE_HEADS * tree_size * tree_size;
    let num_blocks = tree_size.div_ceil(16);
    let a_len = batch_size * VALUE_HEADS * num_blocks * num_blocks.div_ceil(2) * 16 * 32;
    let a_inv_len = batch_size * VALUE_HEADS * num_blocks * 16 * 16;
    let h0_len = batch_size * VALUE_HEADS * HEAD_V_DIM * HEAD_K_DIM;
    let kh0_len = batch_size * tree_size * VALUE_HEADS * HEAD_V_DIM;
    (qk_len * 2 + h0_len) * size_of::<bf16>()
        + batch_size * tree_size * 3 * size_of::<u32>()
        + scalar_len * size_of::<f32>() * 2
        + (out_len + a_len + a_inv_len + kh0_len) * size_of::<f32>()
        + batch_size * size_of::<i32>()
}

#[uzu_bench]
fn bench_build_tree_gram(c: &mut Criterion) {
    let context = <Metal as Backend>::Context::new().expect("metal context");
    let kernel_paths = if context.supports_mxu() {
        &[("Simdgroup", false), ("MXU", true)][..]
    } else {
        &[("Simdgroup", false)][..]
    };

    for &(kernel_path, use_mxu) in kernel_paths {
        let mut group = c.benchmark_group(format!("Metal/Kernel/GDNTreeVerify/BuildTreeGram/{kernel_path}"));
        group.sample_size(20).warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(2));

        for &batch_size in BATCH_SIZES {
            for &tree_size in TREE_SIZES {
                let kernel = <<Metal as Backend>::Kernels as Kernels>::BuildTreeGramKernel::new(
                    &context,
                    DataType::BF16,
                    use_mxu,
                    true,
                )
                .expect("BuildTreeGramKernel");
                let scale = (HEAD_K_DIM as f32).sqrt().recip();
                let benchmark_path =
                    format!("Metal/Kernel/GDNTreeVerify/BuildTreeGram/{kernel_path}/B{batch_size}_T{tree_size}");
                let benchmark_id = BenchmarkId::from_parameter(format!(
                    "B{batch_size}_T{tree_size}_Hg{K_HEADS}_HV{VALUE_HEADS}_K{HEAD_K_DIM}"
                ));

                let mut buffers = ColdPool::new(buffers_bytes(batch_size, tree_size), || {
                    make_buffers(&context, batch_size, tree_size).0
                });
                group.bench_function(benchmark_id, |bencher| {
                    iter_encode_loop_named::<Metal, _>(context.as_ref(), bencher, &benchmark_path, |encoder| {
                        let buffers = buffers.next_mut();
                        kernel.encode(
                            &buffers.q,
                            &buffers.k,
                            &buffers.trie,
                            &buffers.prefix,
                            &buffers.beta,
                            Some(&buffers.h0),
                            Some(&buffers.h0_idx),
                            &mut buffers.a_packed,
                            &mut buffers.qkd,
                            &mut buffers.a_inv,
                            Some(&mut buffers.kh0),
                            scale,
                            batch_size as u32,
                            tree_size as u32,
                            K_HEADS as u32,
                            VALUE_HEADS as u32,
                            HEAD_K_DIM as u32,
                            HEAD_V_DIM as u32,
                            encoder,
                        );
                    });
                });
            }
        }
        group.finish();
    }
}
