#![cfg(metal_backend)]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion};
use half::bf16;
use num_traits::Float;
use proc_macros::uzu_bench;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Allocation, Backend, Context, Kernels, kernel::BuildTreeOutKernel},
        metal::Metal,
    },
    tests::{
        helpers::{alloc_allocation, alloc_allocation_with_data},
        matmul::iter_encode_loop_named,
    },
};

const QK_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const HEAD_K_DIM: usize = 128;
const HEAD_V_DIM: usize = 128;
const TREE_OUT_MATMUL_MXU_ROWS: usize = 16;
const TREE_OUT_MATMUL_SIMDGROUP_ROWS: usize = 8;
const TREE_SIZES: &[usize] = &[32, 33, 49, 64, 128, 256, 512];
const BATCH_SIZES: &[usize] = &[1, 2, 4, 8];
const TREE_OUT_VALUE_COLS: usize = 32;
const TREE_OUT_SIMDGROUPS_PER_TG: usize = 4;
const BUILD_TREE_OUT_PATHS: &[(&str, bool, bool, usize)] = &[
    ("Simdgroup/H0Direct", false, false, TREE_OUT_MATMUL_SIMDGROUP_ROWS),
    ("Simdgroup/H0Transposed", false, true, TREE_OUT_MATMUL_SIMDGROUP_ROWS),
    ("MXU/H0Direct", true, false, TREE_OUT_MATMUL_MXU_ROWS),
];

struct TreeOutBuffers {
    q: Allocation<Metal>,
    prefix: Allocation<Metal>,
    qkd: Allocation<Metal>,
    u: Allocation<Metal>,
    h0: Allocation<Metal>,
    h0_indices: Allocation<Metal>,
    o: Allocation<Metal>,
}

fn make_buffers<T: ArrayElement + Float>(
    context: &<Metal as Backend>::Context,
    batch_size: usize,
    tree_size: usize,
) -> TreeOutBuffers {
    let q_len = batch_size * tree_size * QK_HEADS * HEAD_K_DIM;
    let prefix_len = batch_size * tree_size * VALUE_HEADS;
    let qkd_len = batch_size * VALUE_HEADS * tree_size * tree_size;
    let uv_len = batch_size * VALUE_HEADS * tree_size * HEAD_V_DIM;
    let h0_len = batch_size * VALUE_HEADS * HEAD_V_DIM * HEAD_K_DIM;
    let q = (0..q_len).map(|i| T::from(((i as f32 * 0.017).sin() * 0.2) + 0.01).unwrap()).collect::<Vec<_>>();
    let prefix = (0..prefix_len)
        .map(|i| -((i % tree_size) as f32) * 0.01 - ((i % VALUE_HEADS) as f32) * 0.003)
        .collect::<Vec<_>>();
    let qkd = (0..qkd_len).map(|i| ((i as f32 * 0.013).cos() * 0.03) - 0.01).collect::<Vec<_>>();
    let u = (0..uv_len).map(|i| (i as f32 * 0.011).sin() * 0.3 + 0.04).collect::<Vec<_>>();
    let h0 = (0..h0_len).map(|i| ((i as f32 * 0.019).cos() * 0.2) - 0.01).collect::<Vec<_>>();
    let h0_indices = (0..batch_size as i32).collect::<Vec<_>>();

    TreeOutBuffers {
        q: alloc_allocation_with_data::<Metal, T>(context, &q),
        prefix: alloc_allocation_with_data::<Metal, f32>(context, &prefix),
        qkd: alloc_allocation_with_data::<Metal, f32>(context, &qkd),
        u: alloc_allocation_with_data::<Metal, f32>(context, &u),
        h0: alloc_allocation_with_data::<Metal, f32>(context, &h0),
        h0_indices: alloc_allocation_with_data::<Metal, i32>(context, &h0_indices),
        o: alloc_allocation::<Metal, T>(context, uv_len),
    }
}

#[uzu_bench]
fn bench_build_tree_out(c: &mut Criterion) {
    let context = <Metal as Backend>::Context::new().expect("metal context");
    bench_build_tree_out_type::<bf16>(c, &context, "BF16");
    bench_build_tree_out_type::<f32>(c, &context, "F32");
}

fn bench_build_tree_out_type<T: ArrayElement + Float>(
    c: &mut Criterion,
    context: &<Metal as Backend>::Context,
    data_type_name: &str,
) {
    for &(path, use_mxu, transposed_h0, tile_rows) in BUILD_TREE_OUT_PATHS {
        if use_mxu && !context.supports_mxu() {
            continue;
        }
        let tile_name = format!("Rows{tile_rows}_VCols{TREE_OUT_VALUE_COLS}_SG{TREE_OUT_SIMDGROUPS_PER_TG}");
        let mut group =
            c.benchmark_group(format!("Metal/Kernel/GDNTreeVerify/BuildTreeOut/{data_type_name}/{path}/{tile_name}"));
        group.sample_size(10).warm_up_time(Duration::from_millis(100)).measurement_time(Duration::from_millis(500));

        for &batch_size in BATCH_SIZES {
            for &tree_size in TREE_SIZES {
                let mut buffers = make_buffers::<T>(context, batch_size, tree_size);
                let benchmark_path = format!(
                    "Metal/Kernel/GDNTreeVerify/BuildTreeOut/{data_type_name}/{path}/{tile_name}/B{batch_size}_T{tree_size}"
                );
                let benchmark_id = BenchmarkId::from_parameter(format!(
                    "B{batch_size}_T{tree_size}_QK{QK_HEADS}_HV{VALUE_HEADS}_K{HEAD_K_DIM}_V{HEAD_V_DIM}"
                ));

                let kernel = <<Metal as Backend>::Kernels as Kernels>::BuildTreeOutKernel::new(
                    context,
                    T::data_type(),
                    T::data_type(),
                    use_mxu,
                    transposed_h0,
                    true,
                )
                .expect("BuildTreeOutKernel");
                group.bench_function(benchmark_id, |bencher| {
                    iter_encode_loop_named::<Metal, _>(context, bencher, &benchmark_path, |encoder| {
                        kernel.encode(
                            &buffers.q,
                            &buffers.prefix,
                            &buffers.qkd,
                            &buffers.u,
                            Some(&buffers.h0),
                            Some(&buffers.h0_indices),
                            &mut buffers.o,
                            (HEAD_K_DIM as f32).sqrt().recip(),
                            batch_size as u32,
                            tree_size as u32,
                            QK_HEADS as u32,
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
