#![cfg(metal_backend)]

use std::mem::size_of;

use criterion::{BenchmarkId, Criterion, Throughput};
use proc_macros::uzu_bench;

use crate::{
    backends::{
        common::{Allocation, Backend, Kernels, kernel::BuildTreePrefixKernel},
        metal::Metal,
    },
    tests::{
        cold_pool::ColdPool,
        helpers::{alloc_allocation, alloc_allocation_with_data},
        matmul::iter_encode_loop_named,
        util::type_short_name,
    },
};

struct TreePrefixBuffers {
    trie: Allocation<Metal>,
    log_decay: Allocation<Metal>,
    prefix: Allocation<Metal>,
}

#[uzu_bench]
fn bench_build_tree_prefix(c: &mut Criterion) {
    let context = crate::tests::util::shared_metal_context();
    let kernel =
        <<Metal as Backend>::Kernels as Kernels>::BuildTreePrefixKernel::new(&context).expect("BuildTreePrefixKernel");
    let mut group = c.benchmark_group(format!("{}/Kernel/GDNTreeVerify/BuildTreePrefix", type_short_name::<Metal>()));

    for tree_size in [49usize, 64, 128] {
        let batch_size = 1usize;
        let value_heads = 48usize;
        let len = batch_size * tree_size * value_heads;
        let trie = (0..tree_size as u32).flat_map(|row| [row, tree_size as u32 - 1, row]).collect::<Vec<_>>();
        let log_decay = (0..len).map(|i| -0.001 - (i as f32 * 0.017).sin().abs() * 0.1).collect::<Vec<_>>();
        let bytes_per_copy = trie.len() * size_of::<u32>() + len * size_of::<f32>() * 2;
        let mut buffers = ColdPool::new(bytes_per_copy, || TreePrefixBuffers {
            trie: alloc_allocation_with_data::<Metal, u32>(&context, &trie),
            log_decay: alloc_allocation_with_data::<Metal, f32>(&context, &log_decay),
            prefix: alloc_allocation::<Metal, f32>(&context, len),
        });

        group.throughput(Throughput::Elements((len * tree_size) as u64));
        group.bench_function(BenchmarkId::from_parameter(format!("B1_T{tree_size}_HV{value_heads}")), |bencher| {
            let benchmark_path = format!(
                "{}/Kernel/GDNTreeVerify/BuildTreePrefix/B1_T{tree_size}_HV{value_heads}",
                type_short_name::<Metal>()
            );
            iter_encode_loop_named::<Metal, _>(&context, bencher, &benchmark_path, |encoder| {
                let buffers = buffers.next_mut();
                kernel.encode(
                    &buffers.trie,
                    &buffers.log_decay,
                    &mut buffers.prefix,
                    batch_size as u32,
                    tree_size as u32,
                    value_heads as u32,
                    encoder,
                );
            });
        });
    }
}
