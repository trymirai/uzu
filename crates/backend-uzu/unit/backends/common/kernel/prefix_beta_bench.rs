#![cfg(metal_backend)]

use std::mem::size_of;

use criterion::{BenchmarkId, Criterion, Throughput};
use proc_macros::uzu_bench;

use crate::{
    array::ArrayContextExt,
    backends::{
        common::{Allocation, Backend, Kernels, kernel::BuildPrefixBetaKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::{cold_pool::ColdPool, matmul::iter_encode_loop_named, util::type_short_name},
};

fn chain_path_matrix(tree_size: usize) -> Vec<u8> {
    let mut path_matrix = vec![0; tree_size * tree_size];
    for row in 0..tree_size {
        for col in 0..=row {
            path_matrix[row * tree_size + col] = 1;
        }
    }
    path_matrix
}

struct PrefixBetaBuffers {
    path_matrix: Allocation<Metal>,
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

    for tree_size in [49usize, 129] {
        let batch_size = 1usize;
        let value_heads = 48usize;
        let len = batch_size * tree_size * value_heads;
        let path_matrix = chain_path_matrix(tree_size);
        let a_transposed = (0..len).map(|i| (i as f32 * 0.17).sin() - 0.4).collect::<Vec<_>>();
        let b_data = (0..len).map(|i| (i as f32 * 0.11).cos() * 0.8).collect::<Vec<_>>();
        let a_log = (0..value_heads).map(|i| -0.2 + i as f32 * 0.03).collect::<Vec<_>>();
        let dt_bias = (0..value_heads).map(|i| 0.1 - i as f32 * 0.02).collect::<Vec<_>>();

        let a_log = context.create_array_from(&[a_log.len()], &a_log);
        let dt_bias = context.create_array_from(&[dt_bias.len()], &dt_bias);
        let bytes_per_copy = path_matrix.len() + len * size_of::<f32>() * 4;
        let mut buffers = ColdPool::new(bytes_per_copy, || PrefixBetaBuffers {
            path_matrix: context.create_array_from(&[path_matrix.len()], &path_matrix).into_allocation(),
            a_transposed: context.create_array_from(&[a_transposed.len()], &a_transposed).into_allocation(),
            b: context.create_array_from(&[b_data.len()], &b_data).into_allocation(),
            prefix: context.create_array_uninitialized(&[len], DataType::F32).into_allocation(),
            beta: context.create_array_uninitialized(&[len], DataType::F32).into_allocation(),
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
                    &buffers.path_matrix,
                    &buffers.a_transposed,
                    &buffers.b,
                    a_log.allocation(),
                    dt_bias.allocation(),
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
