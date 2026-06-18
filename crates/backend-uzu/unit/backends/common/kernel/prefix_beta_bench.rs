#![cfg(metal_backend)]

use criterion::{BenchmarkId, Criterion, Throughput};
use proc_macros::uzu_bench;

use crate::{
    array::ArrayContextExt,
    backends::{
        common::{Backend, Kernels, kernel::BuildPrefixBetaKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::{matmul::iter_encode_loop_named, util::type_short_name},
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
        let a = (0..len).map(|i| (i as f32 * 0.17).sin() - 0.4).collect::<Vec<_>>();
        let b_data = (0..len).map(|i| (i as f32 * 0.11).cos() * 0.8).collect::<Vec<_>>();
        let a_log = (0..value_heads).map(|i| -0.2 + i as f32 * 0.03).collect::<Vec<_>>();
        let dt_bias = (0..value_heads).map(|i| 0.1 - i as f32 * 0.02).collect::<Vec<_>>();

        let path_matrix = context.create_array_from(&[path_matrix.len()], &path_matrix);
        let a = context.create_array_from(&[a.len()], &a);
        let b_data = context.create_array_from(&[b_data.len()], &b_data);
        let a_log = context.create_array_from(&[a_log.len()], &a_log);
        let dt_bias = context.create_array_from(&[dt_bias.len()], &dt_bias);
        let mut prefix = context.create_array_uninitialized(&[len], DataType::F32).into_allocation();
        let mut beta = context.create_array_uninitialized(&[len], DataType::F32).into_allocation();

        group.throughput(Throughput::Elements((len * tree_size) as u64));
        group.bench_function(BenchmarkId::from_parameter(format!("B1_T{tree_size}_HV{value_heads}")), |bencher| {
            let benchmark_path = format!(
                "{}/Kernel/GDNTreeVerify/BuildPrefixBeta/B1_T{tree_size}_HV{value_heads}",
                type_short_name::<Metal>()
            );
            iter_encode_loop_named::<Metal, _>(&context, bencher, &benchmark_path, |encoder| {
                kernel.encode(
                    path_matrix.allocation(),
                    a.allocation(),
                    b_data.allocation(),
                    a_log.allocation(),
                    dt_bias.allocation(),
                    &mut prefix,
                    &mut beta,
                    batch_size as u32,
                    tree_size as u32,
                    value_heads as u32,
                    encoder,
                );
            });
        });
    }
}
