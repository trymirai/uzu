#![cfg(metal_backend)]

use std::{mem::size_of, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    array::ArrayContextExt,
    backends::{
        common::{Allocation, Backend, Context, Encoder, Kernels, kernel::GdnTreeUpdateSolveKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::{cold_pool::ColdPool, matmul::iter_encode_loop_named},
};

const NUM_K_HEADS: usize = 16;
const NUM_V_HEADS: usize = 48;
const HEAD_K_DIM: usize = 128;
const HEAD_V_DIM: usize = 128;
const BT: usize = 16;
const BV: usize = 16;
const USE_L2NORM: bool = true;

const BENCH_SHAPES: &[(usize, usize)] = &[
    (1, 33),
    (1, 49),
    (1, 64),
    (1, 128),
    (1, 256),
    (1, 512),
    (2, 33),
    (2, 49),
    (2, 64),
    (2, 128),
    (2, 256),
    (2, 512),
    (4, 33),
    (4, 49),
    (4, 64),
    (4, 128),
    (4, 256),
    (4, 512),
    (8, 33),
    (8, 49),
    (8, 64),
    (8, 128),
    (8, 256),
    (8, 512),
];

struct TreeUpdateSolveBuffers {
    k: Allocation<Metal>,
    v: Allocation<Metal>,
    prefix: Allocation<Metal>,
    beta: Allocation<Metal>,
    a: Allocation<Metal>,
    a_inv: Allocation<Metal>,
    h0: Allocation<Metal>,
    h0_idx: Allocation<Metal>,
    u: Allocation<Metal>,
}

fn matrix_value(
    batch_head_matrix_idx: usize,
    row: usize,
    col: usize,
    tree_size: usize,
) -> f32 {
    if col < row {
        ((batch_head_matrix_idx * tree_size * tree_size + row * tree_size + col) as f32 * 0.011).sin() * 0.01
    } else {
        0.0
    }
}

fn inverse_value(
    batch_head_matrix_idx: usize,
    row: usize,
    col: usize,
    tree_size: usize,
) -> f32 {
    if row == col {
        1.0
    } else if row / BT == col / BT && col < row {
        -matrix_value(batch_head_matrix_idx, row, col, tree_size)
    } else {
        0.0
    }
}

fn make_buffers(
    context: &<Metal as Backend>::Context,
    batch_size: usize,
    tree_size: usize,
) -> TreeUpdateSolveBuffers {
    let k_len = batch_size * tree_size * NUM_K_HEADS * HEAD_K_DIM;
    let v_len = batch_size * tree_size * NUM_V_HEADS * HEAD_V_DIM;
    let scalar_len = batch_size * tree_size * NUM_V_HEADS;
    let matrix_len = batch_size * NUM_V_HEADS * tree_size * tree_size;
    let h0_len = batch_size * NUM_V_HEADS * HEAD_V_DIM * HEAD_K_DIM;
    let u_len = batch_size * NUM_V_HEADS * tree_size * HEAD_V_DIM;

    let k = (0..k_len).map(|i| bf16::from_f32(((i as f32 * 0.019).sin() * 0.2) + 0.01)).collect::<Vec<_>>();
    let v = (0..v_len).map(|i| bf16::from_f32(((i as f32 * 0.017).cos() * 0.18) - 0.02)).collect::<Vec<_>>();
    let prefix = (0..scalar_len)
        .map(|i| -((i % tree_size) as f32) * 0.01 - ((i % NUM_V_HEADS) as f32) * 0.003)
        .collect::<Vec<_>>();
    let beta = (0..scalar_len).map(|i| 0.25 + ((i as f32 * 0.013).sin() + 1.0) * 0.2).collect::<Vec<_>>();
    let a = (0..matrix_len)
        .map(|i| {
            let col = i % tree_size;
            let row = (i / tree_size) % tree_size;
            let batch_head_matrix_idx = i / (tree_size * tree_size);
            matrix_value(batch_head_matrix_idx, row, col, tree_size)
        })
        .collect::<Vec<_>>();
    let a_inv = (0..matrix_len)
        .map(|i| {
            let col = i % tree_size;
            let row = (i / tree_size) % tree_size;
            let batch_head_matrix_idx = i / (tree_size * tree_size);
            inverse_value(batch_head_matrix_idx, row, col, tree_size)
        })
        .collect::<Vec<_>>();
    let h0 = (0..h0_len).map(|i| bf16::from_f32(((i as f32 * 0.007).sin() * 0.05) - 0.01)).collect::<Vec<_>>();
    let h0_idx = (0..batch_size).map(|i| i as i32).collect::<Vec<_>>();

    TreeUpdateSolveBuffers {
        k: context.create_array_from(&[k.len()], &k).into_allocation(),
        v: context.create_array_from(&[v.len()], &v).into_allocation(),
        prefix: context.create_array_from(&[prefix.len()], &prefix).into_allocation(),
        beta: context.create_array_from(&[beta.len()], &beta).into_allocation(),
        a: context.create_array_from(&[a.len()], &a).into_allocation(),
        a_inv: context.create_array_from(&[a_inv.len()], &a_inv).into_allocation(),
        h0: context.create_array_from(&[h0.len()], &h0).into_allocation(),
        h0_idx: context.create_array_from(&[h0_idx.len()], &h0_idx).into_allocation(),
        u: context.create_array_uninitialized(&[u_len], DataType::F32).into_allocation(),
    }
}

fn buffers_bytes(
    batch_size: usize,
    tree_size: usize,
) -> usize {
    let k_len = batch_size * tree_size * NUM_K_HEADS * HEAD_K_DIM;
    let v_len = batch_size * tree_size * NUM_V_HEADS * HEAD_V_DIM;
    let scalar_len = batch_size * tree_size * NUM_V_HEADS;
    let matrix_len = batch_size * NUM_V_HEADS * tree_size * tree_size;
    let h0_len = batch_size * NUM_V_HEADS * HEAD_V_DIM * HEAD_K_DIM;
    let u_len = batch_size * NUM_V_HEADS * tree_size * HEAD_V_DIM;

    (k_len + v_len + h0_len) * size_of::<bf16>()
        + (scalar_len * 2 + matrix_len * 2 + u_len) * size_of::<f32>()
        + batch_size * size_of::<i32>()
}

#[uzu_bench]
fn bench_gdn_tree_update_solve(c: &mut Criterion) {
    let context = <Metal as Backend>::Context::new().expect("metal context");
    let kernel = <<Metal as Backend>::Kernels as Kernels>::GdnTreeUpdateSolveKernel::new(
        &context,
        DataType::BF16,
        HEAD_K_DIM as u32,
        BT as u32,
        BV as u32,
        USE_L2NORM,
    )
    .expect("GdnTreeUpdateSolveKernel");

    let mut group = c.benchmark_group("Metal/Kernel/GDNTreeVerify/TreeUpdateSolve");
    group.sample_size(10).warm_up_time(Duration::from_millis(100)).measurement_time(Duration::from_millis(500));

    for &(batch_size, tree_size) in BENCH_SHAPES {
        let benchmark_path = format!("Metal/Kernel/GDNTreeVerify/TreeUpdateSolve/B{batch_size}_T{tree_size}");
        let benchmark_id = BenchmarkId::from_parameter(format!(
            "B{batch_size}_T{tree_size}_Hg{NUM_K_HEADS}_HV{NUM_V_HEADS}_K{HEAD_K_DIM}_V{HEAD_V_DIM}_BV{BV}"
        ));
        let mut buffers =
            ColdPool::new(buffers_bytes(batch_size, tree_size), || make_buffers(&context, batch_size, tree_size));

        group.throughput(Throughput::Elements((batch_size * tree_size * NUM_V_HEADS * HEAD_V_DIM) as u64));
        group.bench_function(benchmark_id, |bencher| {
            iter_encode_loop_named::<Metal, _>(context.as_ref(), bencher, &benchmark_path, |encoder| {
                let buffers = buffers.next_mut();
                encode(&kernel, buffers, batch_size, tree_size, encoder);
            });
        });
    }

    group.finish();
}

fn encode(
    kernel: &impl GdnTreeUpdateSolveKernel<Backend = Metal>,
    buffers: &mut TreeUpdateSolveBuffers,
    batch_size: usize,
    tree_size: usize,
    encoder: &mut Encoder<Metal>,
) {
    kernel.encode(
        &buffers.k,
        &buffers.v,
        &buffers.prefix,
        &buffers.beta,
        &buffers.a,
        &buffers.a_inv,
        &buffers.h0,
        &buffers.h0_idx,
        &mut buffers.u,
        batch_size as u32,
        tree_size as u32,
        NUM_V_HEADS as u32,
        NUM_K_HEADS as u32,
        HEAD_V_DIM as u32,
        encoder,
    );
}
