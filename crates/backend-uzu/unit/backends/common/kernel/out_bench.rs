#![cfg(metal_backend)]

use std::{mem::size_of, time::Duration};

use criterion::{BenchmarkId, Criterion};
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    array::ArrayContextExt,
    backends::{
        common::{Allocation, Backend, Context, Kernels, kernel::BuildTreeOutOutputTileMatmulDirectKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::matmul::iter_encode_loop_named,
};

const QK_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const HEAD_K_DIM: usize = 128;
const HEAD_V_DIM: usize = 128;
const TREE_SIZES: &[usize] = &[33, 49, 64, 128, 256, 512];
const BATCH_SIZES: &[usize] = &[1, 2, 4, 8];

#[derive(Clone, Copy)]
enum Candidate {
    OutputTileMatmulDirectSimdgroup,
    OutputTileMatmulDirectMxu,
}

struct TreeOutBuffers {
    q: Allocation<Metal>,
    prefix: Allocation<Metal>,
    qkd: Allocation<Metal>,
    u: Allocation<Metal>,
    h0: Allocation<Metal>,
    h0_indices: Allocation<Metal>,
    o: Allocation<Metal>,
}

fn make_buffers(
    context: &<Metal as Backend>::Context,
    batch_size: usize,
    tree_size: usize,
) -> TreeOutBuffers {
    let q_len = batch_size * tree_size * QK_HEADS * HEAD_K_DIM;
    let prefix_len = batch_size * tree_size * VALUE_HEADS;
    let qkd_len = batch_size * VALUE_HEADS * tree_size * tree_size;
    let uv_len = batch_size * VALUE_HEADS * tree_size * HEAD_V_DIM;
    let h0_len = batch_size * VALUE_HEADS * HEAD_V_DIM * HEAD_K_DIM;
    let q = (0..q_len).map(|i| bf16::from_f32(((i as f32 * 0.017).sin() * 0.2) + 0.01)).collect::<Vec<_>>();
    let prefix = (0..prefix_len)
        .map(|i| -((i % tree_size) as f32) * 0.01 - ((i % VALUE_HEADS) as f32) * 0.003)
        .collect::<Vec<_>>();
    let qkd = (0..qkd_len).map(|i| ((i as f32 * 0.013).cos() * 0.03) - 0.01).collect::<Vec<_>>();
    let u = (0..uv_len).map(|i| bf16::from_f32(((i as f32 * 0.011).sin() * 0.3) + 0.04)).collect::<Vec<_>>();
    let h0 = (0..h0_len).map(|i| bf16::from_f32(((i as f32 * 0.019).cos() * 0.2) - 0.01)).collect::<Vec<_>>();
    let h0_indices = (0..batch_size as i32).collect::<Vec<_>>();

    TreeOutBuffers {
        q: context.create_array_from(&[q.len()], &q).into_allocation(),
        prefix: context.create_array_from(&[prefix.len()], &prefix).into_allocation(),
        qkd: context.create_array_from(&[qkd.len()], &qkd).into_allocation(),
        u: context.create_array_from(&[u.len()], &u).into_allocation(),
        h0: context.create_array_from(&[h0.len()], &h0).into_allocation(),
        h0_indices: context.create_array_from(&[h0_indices.len()], &h0_indices).into_allocation(),
        o: context.create_array_uninitialized(&[uv_len], DataType::BF16).into_allocation(),
    }
}

fn buffers_bytes(
    batch_size: usize,
    tree_size: usize,
) -> usize {
    let q_len = batch_size * tree_size * QK_HEADS * HEAD_K_DIM;
    let prefix_len = batch_size * tree_size * VALUE_HEADS;
    let qkd_len = batch_size * VALUE_HEADS * tree_size * tree_size;
    let uv_len = batch_size * VALUE_HEADS * tree_size * HEAD_V_DIM;
    let h0_len = batch_size * VALUE_HEADS * HEAD_V_DIM * HEAD_K_DIM;
    q_len * size_of::<bf16>()
        + prefix_len * size_of::<f32>()
        + qkd_len * size_of::<f32>()
        + uv_len * size_of::<bf16>() * 2
        + h0_len * size_of::<bf16>()
        + batch_size * size_of::<i32>()
}

fn flops(
    batch_size: usize,
    tree_size: usize,
) -> usize {
    2 * batch_size * VALUE_HEADS * tree_size * HEAD_V_DIM * (HEAD_K_DIM + tree_size)
}

fn candidate_name(candidate: Candidate) -> &'static str {
    match candidate {
        Candidate::OutputTileMatmulDirectSimdgroup => "OutputTileMatmulDirectSimdgroup",
        Candidate::OutputTileMatmulDirectMxu => "OutputTileMatmulDirectMxu",
    }
}

#[uzu_bench]
fn bench_build_tree_out(c: &mut Criterion) {
    let context = <Metal as Backend>::Context::new().expect("metal context");
    let candidates = [Candidate::OutputTileMatmulDirectSimdgroup, Candidate::OutputTileMatmulDirectMxu];

    for candidate in candidates {
        if matches!(candidate, Candidate::OutputTileMatmulDirectMxu) && !context.supports_mxu() {
            continue;
        }

        let candidate_name = candidate_name(candidate);
        let mut group = c.benchmark_group(format!("Metal/Kernel/GDNTreeVerify/BuildTreeOut/{candidate_name}"));
        group.sample_size(10).warm_up_time(Duration::from_millis(100)).measurement_time(Duration::from_millis(500));

        for &batch_size in BATCH_SIZES {
            for &tree_size in TREE_SIZES {
                let mut buffers = make_buffers(&context, batch_size, tree_size);
                let benchmark_path =
                    format!("Metal/Kernel/GDNTreeVerify/BuildTreeOut/{candidate_name}/B{batch_size}_T{tree_size}");
                let benchmark_id = BenchmarkId::from_parameter(format!(
                    "B{batch_size}_T{tree_size}_QK{QK_HEADS}_HV{VALUE_HEADS}_K{HEAD_K_DIM}_V{HEAD_V_DIM}_flops{}_bytes{}",
                    flops(batch_size, tree_size),
                    buffers_bytes(batch_size, tree_size)
                ));

                let use_mxu = matches!(candidate, Candidate::OutputTileMatmulDirectMxu);
                let kernel = <<Metal as Backend>::Kernels as Kernels>::BuildTreeOutOutputTileMatmulDirectKernel::new(
                    &context,
                    DataType::BF16,
                    true,
                    use_mxu,
                )
                .expect("BuildTreeOutOutputTileMatmulDirectKernel");
                group.bench_function(benchmark_id, |bencher| {
                    iter_encode_loop_named::<Metal, _>(context.as_ref(), bencher, &benchmark_path, |encoder| {
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
