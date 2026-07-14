#![cfg(metal_backend)]

use std::mem::size_of;

use criterion::{BenchmarkId, Criterion, Throughput};
use proc_macros::uzu_bench;

use crate::{
    backends::{
        common::{Allocation, Backend, Kernels, kernel::BuildPrefixBetaKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::{
        cold_pool::ColdPool,
        helpers::{alloc_allocation, alloc_allocation_with_data},
        matmul::iter_encode_loop_named,
        util::type_short_name,
    },
};

// chain trie: node `row` covers [row, T-1].
fn chain_trie(tree_size: usize) -> Vec<u32> {
    let mut trie = Vec::with_capacity(tree_size * 3);
    for row in 0..tree_size as u32 {
        trie.extend_from_slice(&[row, tree_size as u32 - 1, row]);
    }
    trie
}

struct PrefixBetaBuffers {
    trie: Allocation<Metal>,
    a_transposed: Allocation<Metal>,
    b: Allocation<Metal>,
    prefix: Allocation<Metal>,
    beta: Allocation<Metal>,
}

#[uzu_bench]
fn bench_build_prefix_beta(c: &mut Criterion) {
    let context = crate::tests::util::shared_metal_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::BuildPrefixBetaKernel::new(&context, DataType::F32)
        .expect("BuildPrefixBetaKernel");

    let mut group = c.benchmark_group(format!("{}/Kernel/GDNTreeVerify/BuildPrefixBeta", type_short_name::<Metal>()));

    for tree_size in [49usize, 64, 129] {
        let batch_size = 1usize;
        let value_heads = 48usize;
        let len = batch_size * tree_size * value_heads;
        let trie = chain_trie(tree_size);
        let a_transposed = (0..len).map(|i| (i as f32 * 0.17).sin() - 0.4).collect::<Vec<_>>();
        let b_data = (0..len).map(|i| (i as f32 * 0.11).cos() * 0.8).collect::<Vec<_>>();
        let a_log = (0..value_heads).map(|i| -0.2 + i as f32 * 0.03).collect::<Vec<_>>();
        let dt_bias = (0..value_heads).map(|i| 0.1 - i as f32 * 0.02).collect::<Vec<_>>();

        let a_log = alloc_allocation_with_data::<Metal, f32>(&context, &a_log);
        let dt_bias = alloc_allocation_with_data::<Metal, f32>(&context, &dt_bias);
        let bytes_per_copy = trie.len() * size_of::<u32>() + len * size_of::<f32>() * 4;
        let mut buffers = ColdPool::new(bytes_per_copy, || PrefixBetaBuffers {
            trie: alloc_allocation_with_data::<Metal, u32>(&context, &trie),
            a_transposed: alloc_allocation_with_data::<Metal, f32>(&context, &a_transposed),
            b: alloc_allocation_with_data::<Metal, f32>(&context, &b_data),
            prefix: alloc_allocation::<Metal, f32>(&context, len),
            beta: alloc_allocation::<Metal, f32>(&context, len),
        });

        group.throughput(Throughput::Elements((len * tree_size) as u64));
        group.bench_function(BenchmarkId::from_parameter(format!("B1_T{tree_size}_HV{value_heads}")), |bencher| {
            let benchmark_path = format!(
                "{}/Kernel/GDNTreeVerify/BuildPrefixBeta/B1_T{tree_size}_HV{value_heads}",
                type_short_name::<Metal>()
            );
            iter_encode_loop_named::<Metal, _>(&context, bencher, &benchmark_path, |encoder| {
                let buffers = buffers.next_mut();
                kernel.encode(
                    &buffers.trie,
                    &buffers.a_transposed,
                    &buffers.b,
                    &a_log,
                    &dt_bias,
                    &mut buffers.prefix,
                    &mut buffers.beta,
                    batch_size as u32,
                    tree_size as u32,
                    value_heads as u32,
                    encoder,
                );
            });
        });
    }
}
